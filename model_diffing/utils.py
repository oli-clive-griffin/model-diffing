import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import einops
import torch
import yaml  # type: ignore
from einops.einops import Reduction
from pydantic import BaseModel as _BaseModel
from torch import nn

from model_diffing.log import logger

python_version = sys.version_info
if python_version.major == 3 and python_version.minor < 11:
    Self = Any
else:
    from typing import Self


class BaseModel(_BaseModel):
    class Config:
        extra = "forbid"


class SaveableModule(nn.Module, ABC):
    @abstractmethod
    def _dump_cfg(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def _from_cfg(cls: type[Self], cfg: dict[str, Any]) -> Self: ...

    def save(self, basepath: Path):
        basepath.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), basepath / "model.pt")
        with open(basepath / "model_cfg.yaml", "w") as f:
            yaml.dump(self._dump_cfg(), f)

    @classmethod
    def load(cls: type[Self], basepath: Path | str, device: torch.device | str = "cpu") -> Self:
        basepath = Path(basepath)
        with open(basepath / "model_cfg.yaml") as f:
            cfg = yaml.safe_load(f)
        model = cls._from_cfg(cfg)
        model.load_state_dict(torch.load(basepath / "model.pt", weights_only=True, map_location=device))
        return model


# Add a custom constructor for the !!python/tuple tag,
# converting the loaded sequence to a Python tuple.
def _tuple_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple_constructor)


# might seem strange to redefine these but:
# 1: these signatures allow us to use these norm functions in einops.reduce
# 2: I (oli) find `l2_norm(x, dim=-1)` more readable than `x.norm(p=2, dim=-1)`


def l0_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=0, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def l1_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=1, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def l2_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=2, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def weighted_l1_sparsity_loss(
    W_dec_HTMPD: torch.Tensor,
    hidden_BH: torch.Tensor,
    hookpoint_reduction: Reduction,  # type: ignore
    model_reduction: Reduction,  # type: ignore
    token_reduction: Reduction,  # type: ignore
) -> torch.Tensor:
    assert (hidden_BH >= 0).all()
    # think about it like: each latent (called "hidden" here) has a separate projection onto each (model, hookpoint)
    # so we have a separate l2 norm for each (hidden, model, hookpoint)
    W_dec_l2_norms_HTMP = einops.reduce(
        W_dec_HTMPD, "hidden token model hookpoint dim -> hidden token model hookpoint", l2_norm
    )

    # to get the weighting factor for each latent, we reduce it's decoder norms for each (model, hookpoint)
    reduced_norms_H = multi_reduce(
        W_dec_l2_norms_HTMP,
        "hidden token model hookpoint",
        ("token", token_reduction),
        ("hookpoint", hookpoint_reduction),
        ("model", model_reduction),
    )

    # now we weight the latents by the sum of their norms
    weighted_hiddens_BH = hidden_BH * reduced_norms_H
    weighted_l1_of_hiddens_BH = einops.reduce(weighted_hiddens_BH, "batch hidden -> batch", l1_norm)
    return weighted_l1_of_hiddens_BH.mean()


sparsity_loss_l2_of_norms = partial(
    weighted_l1_sparsity_loss,
    token_reduction=l2_norm,
    hookpoint_reduction=l2_norm,
    model_reduction=l2_norm,
)

sparsity_loss_l1_of_norms = partial(
    weighted_l1_sparsity_loss,
    token_reduction=l1_norm,
    hookpoint_reduction=l1_norm,
    model_reduction=l1_norm,
)


def calculate_reconstruction_loss_summed_MSEs(activation_BXD: torch.Tensor, target_BXD: torch.Tensor) -> torch.Tensor:
    """This is a little weird because we have both model and hookpoint (aka layer) dimensions, so it's worth explaining deeply:

    The reconstruction loss is a sum of squared L2 norms of the error for each activation space being reconstructed.
    In the Anthropic crosscoders update, they don't write for the multiple-model case, they write it as:

    (using l here for layer, hookpoint is technically more correct):
    $$\\sum_{l \\in L} \\|a^l(x_j) - a^{l'}(x_j)\\|^2$$

    Here, I'm assuming we want to expand that sum to be over models, so we would have:

    $$ \\sum_{m \\in M} \\sum_{l \\in L} \\|a_m^l(x_j) - a_m^{l'}(x_j)\\|^2 $$
    """
    # take the L2 norm of the error inside each d_model feature space
    error_BXD = activation_BXD - target_BXD
    error_norm_BX = einops.reduce(error_BXD, "batch ... d_model -> batch ...", l2_norm)
    squared_error_norm_BX = error_norm_BX.square()

    # sum errors across all crosscoding dimensions
    summed_squared_error_norm_B = einops.reduce(squared_error_norm_BX, "batch ... -> batch", torch.sum)
    return summed_squared_error_norm_B.mean()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# (oli) sorry - this is probably overengineered
def multi_reduce(
    tensor: torch.Tensor,
    shape_pattern: str,
    *reductions: tuple[str, Reduction],  # type: ignore
) -> torch.Tensor:
    original_shape = einops.parse_shape(tensor, shape_pattern)
    for reduction_dim, reduction_fn in reductions:
        if reduction_dim not in original_shape:
            raise ValueError(f"Dimension {reduction_dim} not found in original_shape {original_shape}")
        target_pattern_pattern = shape_pattern.replace(reduction_dim, "")
        exec_pattern = f"{shape_pattern} -> {target_pattern_pattern}"
        shape_pattern = target_pattern_pattern
        tensor = einops.reduce(tensor, exec_pattern, reduction_fn)

    return tensor


def calculate_fvu_X_old(
    activations_BXD: torch.Tensor,
    reconstructed_BXD: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """for each model and hookpoint, calculate the mean fvu inside each d_model feature space"""
    error_BXD = activations_BXD - reconstructed_BXD

    mean_error_var_X = error_BXD.var(-1).mean(0)
    mean_activations_var_X = activations_BXD.var(-1).mean(0)

    return mean_error_var_X / (mean_activations_var_X + eps)


def calculate_vector_norm_fvu_X(
    y_BXD: torch.Tensor,
    y_pred_BXD: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    For each crosscoding output space (model, hookpoint, token, etc.) Calculates the fvu, returning
    a tensor of shape (crosscoding_dim_1, crosscoding_dim_2, ...) where each element is the fvu for the corresponding
    crosscoding output space.

    see https://www.lesswrong.com/posts/ZBjhp6zwfE8o8yfni/#Rm8xDeB95fb2usorb for a discussion of this
    """
    batch_size = y_BXD.shape[0]
    if batch_size == 1:
        logger.warn("Batch size is 1, fvu is not meaningful. returning 0")
        return torch.zeros(y_BXD.shape[1:-1])

    y_mean_BXD = y_BXD.mean(dim=0, keepdim=True)

    var_err_BX = (y_BXD - y_pred_BXD).norm(p=2, dim=-1).square()  # variance
    var_err_X = var_err_BX.mean(0)  # mean over batch

    var_total_BX = (y_BXD - y_mean_BXD).norm(p=2, dim=-1).square()
    var_total_X = var_total_BX.mean(0)  # mean over batch

    return var_err_X / (var_total_X + eps)


def get_fvu_dict(
    y_BXD: torch.Tensor,
    y_pred_BXD: torch.Tensor,
    *crosscoding_dims: tuple[str, list[str] | list[int]],
) -> dict[str, float]:
    """
    crosscoding_dims is a list of tuples, each (a, b) tuple is:
        a: the name of the crosscoding dimension ('hookpoint', 'model', 'token', etc.)
        b: the labels of the crosscoding dimension (e.g. [0, 1, 7] or ['gpt2', 'gpt3', 'gpt4'], or ['<bos>', '-1', 'self'])

    the reason we need the explicit naming pattern is that often indices are not helpful. For example, when training
    a crosscoder on hookpoints 2, 5, and 8, you don't to want to have them labeled [0, 1, 2]. i.e. you need to know what
    each index means.
    """
    fvu_X = calculate_vector_norm_fvu_X(y_BXD, y_pred_BXD)
    assert len(crosscoding_dims) == len(fvu_X.shape)

    # a list of tuples where each tuple is a unique set of indices into the fvu_X tensor
    index_combinations = product(*(range(dim_size) for dim_size in fvu_X.shape))

    fvu_dict = {}
    for indices in index_combinations:
        name = "train/fvu"
        # nest so that wandb puts them all in one graph
        for (dim_name, dim_labels), dim_index in zip(crosscoding_dims, indices, strict=True):
            name += f"/{dim_name}{dim_labels[dim_index]}"

        fvu_dict[name] = fvu_X[indices].item()

    return fvu_dict


def get_summed_decoder_norms_H(W_dec_HXD: torch.Tensor) -> torch.Tensor:
    W_dec_l2_norms_HX = einops.reduce(W_dec_HXD, "hidden ... dim -> hidden ...", l2_norm)
    norms_H = einops.reduce(W_dec_l2_norms_HX, "hidden ... -> hidden", torch.sum)
    return norms_H


def size_human_readable(tensor: torch.Tensor) -> str:
    # Calculate the number of bytes in the tensor
    if tensor.nbytes >= 1024**3:
        return f"{tensor.nbytes / (1024**3):.2f} GB"
    elif tensor.nbytes >= 1024**2:
        return f"{tensor.nbytes / (1024**2):.2f} MB"
    elif tensor.nbytes >= 1024:
        return f"{tensor.nbytes / 1024:.2f} KB"
    else:
        return f"{tensor.nbytes} B"


# hacky but useful for debugging
def inspect(tensor: torch.Tensor) -> str:
    return f"{tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={size_human_readable(tensor)}"


def round_up(x: int, to_multiple_of: int) -> int:
    remainder = x % to_multiple_of
    if remainder != 0:
        x = (((x - remainder) // to_multiple_of) + 1) * to_multiple_of
    return x


def torch_batch_iterator(tensor_iterator_X: Iterator[torch.Tensor], yield_batch_size: int) -> Iterator[torch.Tensor]:
    sample_X = next(tensor_iterator_X)
    batch_BX = torch.empty([yield_batch_size, *sample_X.shape], device=sample_X.device, dtype=sample_X.dtype)
    batch_BX[0] = sample_X
    ptr = 1

    for batch_X in tensor_iterator_X:
        if ptr == yield_batch_size:
            yield batch_BX
            ptr = 0

        batch_BX[ptr] = batch_X
        ptr += 1


# admin function for alerting on finishing long-running cells in notebooks
def beep_macos():
    try:
        # Play the system alert sound
        os.system("afplay /System/Library/Sounds/Sosumi.aiff")
    except Exception as e:
        logger.error(f"Failed to play alarm sound: {e}")


def change_batch_size_BX(
    iterator_HX: Iterator[torch.Tensor],
    new_batch_size_B: int,
    yield_final_batch: bool = False,
) -> Iterator[torch.Tensor]:
    # Get a sample tensor to determine shape/dtype/device
    # But we need to handle this first sample as well
    try:
        sample_HX = next(iterator_HX)
    except StopIteration:
        return  # Empty iterator, nothing to yield

    # Pre-allocate output tensor of batch size B
    X = sample_HX.shape[1:]

    out_BX = torch.empty(
        [new_batch_size_B, *X],
        device=sample_HX.device,
        dtype=sample_HX.dtype,
    )
    out_ptr = 0  # Position in output tensor to fill next

    # Helper function to process a tensor and yield complete batches
    def process_tensor(tensor_HX: torch.Tensor):
        nonlocal out_ptr

        input_batch_size = tensor_HX.shape[0]  # Current batch size (which can vary)
        input_batch_ptr = 0  # Position in input tensor

        while input_batch_ptr < input_batch_size:
            output_space_left = new_batch_size_B - out_ptr
            n_elements_left_in_input = input_batch_size - input_batch_ptr
            n_elements_to_copy = min(output_space_left, n_elements_left_in_input)

            # Copy elements to output tensor
            copied_chunk_HX = tensor_HX[input_batch_ptr : input_batch_ptr + n_elements_to_copy]
            out_BX[out_ptr : out_ptr + n_elements_to_copy] = copied_chunk_HX
            out_ptr += n_elements_to_copy
            input_batch_ptr += n_elements_to_copy

            # If output batch is full, yield it
            if out_ptr == new_batch_size_B:
                yield out_BX.clone()  # Clone to avoid reference issues
                out_ptr = 0  # Reset pointer for next batch

    yield from process_tensor(sample_HX)
    for activations_HX in iterator_HX:
        yield from process_tensor(activations_HX)

    if out_ptr > 0 and yield_final_batch:
        final_batch = out_BX[:out_ptr].clone()
        yield final_batch


@torch.no_grad()
def random_direction_init_(tensor: torch.Tensor, norm: float) -> None:
    tensor.normal_()
    tensor.div_(l2_norm(tensor, dim=-1, keepdim=True))
    tensor.mul_(norm)
