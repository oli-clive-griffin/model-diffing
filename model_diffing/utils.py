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
    B: int,
    yield_final_batch: bool = False,
) -> Iterator[torch.Tensor]:
    leftover_BX: torch.Tensor | None = None
    for activations_HX in iterator_HX:
        if leftover_BX is not None:
            activations_HX = torch.cat([leftover_BX, activations_HX], dim=0)
            leftover_BX = None

        n_batches, remaining_examples = divmod(activations_HX.shape[0], B)
        for i in range(0, n_batches):
            batch_BX = activations_HX[i * B : (i + 1) * B].clone()
            assert batch_BX.shape[0] == B
            yield batch_BX

        if remaining_examples > 0:
            # Create a clone to avoid holding a reference to the full tensor
            leftover_BX = activations_HX[-remaining_examples:].clone()

    if leftover_BX is not None and yield_final_batch:
        yield leftover_BX


@torch.no_grad()
def random_direction_init_(tensor: torch.Tensor, norm: float) -> None:
    tensor.normal_()
    tensor.div_(l2_norm(tensor, dim=-1, keepdim=True))
    tensor.mul_(norm)


# from typing import Callable

# import einops

# # import re
# # from functools import cached_property
# import torch
# from torch import Tensor


# # decorator to log the EinChain shape on error
# def log_shape_on_err(func):  # type: ignore
#     def wrapper(self: "EinTensor", *args, **kwargs):  # type: ignore
#         try:
#             return func(self, *args, **kwargs)
#         except Exception as e:
#             print(f"Error in {func.__name__}: {e}")
#             print(f"Shape: {self.current_pattern()}")
#             raise e

#     return wrapper


# def parse_pattern(pattern: str) -> list[str]:
#     """
#     "b c h w" -> ["b", "c", "h", "w"]
#     "b c (h w)" -> ["b", "c", "(h w)"]
#     """
#     groups = re.findall(r"\w+|\(.*?\)", pattern)
#     return groups


# from functools import wraps
# from typing import Any, Callable

# import einops
# import torch
# from torch import Tensor


# def log_shape_on_err(func):  # type: ignore
#     @wraps(func)
#     def wrapper(self: "EinTensor", *args, **kwargs):  # type: ignore
#         try:
#             return func(self, *args, **kwargs)
#         except Exception as e:
#             print(f"Error in {func.__name__}:")
#             print(f"  Self shape: {self.shape}, dims: {self._einshape}")
#             print(f"  Args: {args}")
#             print(f"  Kwargs: {kwargs}")
#             raise e

#     return wrapper


# class EinTensor:
#     def __init__(self, tensor: torch.Tensor, shape: str) -> None:  # type: ignore
#         """
#         Wrap a PyTorch tensor with named dimensions for einops operations.

#         Args:
#             tensor: The PyTorch tensor to wrap
#             dims: List of dimension names or space-separated string of dimension names
#         """
#         self._tensor = tensor
#         self._einshape = einops.parse_shape(tensor, shape)

#     @property
#     def shape(self):
#         return self._tensor.shape

#     def current_pattern(self) -> str:
#         """Get the current dimension pattern as a space-separated string."""
#         return " ".join(self._einshape.keys())

#     @log_shape_on_err
#     def einsum(self, pattern: str, tensor: "torch.Tensor | EinTensor") -> "EinTensor":  # type: ignore
#         """
#         Perform einsum operation using einops.

#         Args:
#             pattern: Einsum pattern, may use 'self' to refer to this tensor's pattern
#             *tensors: Other tensors to use in the einsum operation
#         """
#         # Extract actual tensors if EinTensors are given
#         other = tensor._tensor if isinstance(tensor, EinTensor) else tensor

#         # Replace 'self' with current pattern
#         trans = self._transformation_with_self(pattern)

#         # Extract output dimensions from pattern
#         out_dims = self._out_shape(pattern)

#         # Perform the operation
#         res = einops.einsum(self._tensor, other, trans)

#         return EinTensor(res, out_dims)

#     @log_shape_on_err
#     def rearrange(self, target_pattern: str, **axes_lengths: int) -> "EinTensor":
#         """
#         Rearrange the tensor dimensions using einops.rearrange.

#         Args:
#             target_pattern: Target pattern for dimensions
#             **axes_lengths: Additional axis information for einops
#         """
#         pattern = self._transformation_to(target_pattern)
#         res = einops.rearrange(self._tensor, pattern, **axes_lengths)
#         return EinTensor(res, self._out_shape(target_pattern))

#     @log_shape_on_err
#     def reduce(
#         self,
#         target_pattern: str,
#         reduction: Reduction,  # type: ignore
#         **axes_lengths: int,
#     ) -> "EinTensor":
#         """
#         Reduce dimensions using einops.reduce.

#         Args:
#             target_pattern: Target pattern after reduction
#             reduction: Reduction function or name
#             **axes_lengths: Additional axis information for einops
#         """
#         pattern = self._transformation_to(target_pattern)
#         res = einops.reduce(self._tensor, pattern, reduction, **axes_lengths)
#         return EinTensor(res, self._out_shape(target_pattern))

#     @log_shape_on_err
#     def repeat(self, target_pattern: str, **axes_lengths: int) -> "EinTensor":
#         """
#         Repeat the tensor using einops.repeat.

#         Args:
#             target_pattern: Target pattern after repeating
#             **axes_lengths: Additional axis information for einops
#         """
#         pattern = self._transformation_to(target_pattern)
#         res = einops.repeat(self._tensor, pattern, **axes_lengths)
#         return EinTensor(res, self._out_shape(target_pattern))

#     def _transformation_with_self(self, pattern: str) -> str:
#         """Convert a pattern that contains 'self' to a full einops pattern."""
#         return pattern.replace("self", self.current_pattern())

#     def _transformation_to(self, target_pattern: str) -> str:
#         """Convert a target pattern to a full einops transformation pattern."""
#         return f"{self.current_pattern()} -> {target_pattern}"

#     def _out_shape(self, target_pattern: str) -> str:
#         """
#         Extract destination shape dimensions from a pattern.

#         Args:
#             target_pattern: Pattern that may include an arrow (->) for transformations

#         Returns:
#             List of dimension names for the result
#         """
#         parts = target_pattern.split("->")
#         if len(parts) == 2:
#             # If there's an arrow, take what's after it, otherwise use the whole pattern
#             result_part = parts[-1].strip()
#             return result_part
#         elif len(parts) == 1:
#             return target_pattern
#         else:
#             raise ValueError(f"Invalid pattern: {target_pattern}")

#     def written_as(self, new_pattern: str) -> "EinTensor":
#         """
#         Change the dimension names without changing the tensor.

#         Args:
#             new_pattern: New dimension names as space-separated string or list
#         """
#         if len(einops.parse_shape(self._tensor, new_pattern)) != len(self._einshape):
#             raise ValueError(
#                 f"Number of dimensions in new pattern ({len(new_pattern)}) "
#                 f"does not match tensor dimensions ({len(self._einshape)})"
#             )

#         return EinTensor(self._tensor, new_pattern)

#     def unwrap(self) -> torch.Tensor:
#         """
#         Unwrap and return the underlying PyTorch tensor.

#         Returns:
#             The underlying PyTorch tensor
#         """
#         return self._tensor

#     def __repr__(self) -> str:
#         dims_str = ", ".join(
#             f"{dim_name}={dim}" if not dim_name.isdigit() else str(dim) for dim_name, dim in zip(self._einshape, self.shape)
#         )
#         return f"EinTensor({dims_str})"

#     # # Basic tensor operations
#     # def __add__(self, other: "torch.Tensor | EinTensor") -> "EinTensor":
#     #     if isinstance(other, EinTensor):
#     #         # Check dimension compatibility
#     #         if self._einshape != other.dims:
#     #             raise ValueError(f"Dimension mismatch: {self._einshape} vs {other.dims}")
#     #         return EinTensor(self._tensor + other._tensor, self._einshape)
#     #     return EinTensor(self._tensor + other, self._einshape)

#     # def __mul__(self, other: "torch.Tensor | EinTensor") -> "EinTensor":
#     #     if isinstance(other, EinTensor):
#     #         # Check dimension compatibility
#     #         if self._einshape != other.dims:
#     #             raise ValueError(f"Dimension mismatch: {self._einshape} vs {other.dims}")
#     #         return EinTensor(self._tensor * other._tensor, self._einshape)
#     #     return EinTensor(self._tensor * other, self._einshape)

#     # Add more operations as needed


# if __name__ == "__main__":
#     x_BCHW = torch.randn(2, 3, 4, 8)
#     y_HSC = torch.randn(4, 5, 3)

#     # Test chain operations
#     out = (
#         EinTensor(x_BCHW, shape="b c h w")
#         # .repeat("b c h w 3")  # Repeat to match dimensions
#         .einsum("self, h s c -> b h w c", y_HSC)  # Multiply with hwc tensor
#         .reduce("b h w", "sum")  # Reduce to single dimension
#         .rearrange("b (h w)")  # Reshape
#         .reduce("b", "sum")  # Sum to scalar
#     )

#     print(f"Final result: {out.unwrap().item()}")

#     # Show dimensions at each step for clarity
#     print("\nStep by step:")

#     step0 = EinTensor(x_BCHW, shape="b c h w")
#     print(f"After einops init: {step0}")

#     step1 = step0.repeat("b c h w 3")
#     print(f"After einops repeat: {step1}")

#     step2 = step1.einsum("self, h w c -> b h w c", y_HSC)
#     print(f"After einsum: {step2}")

#     step3 = step2.reduce("b h w", "sum")
#     print(f"After reduce: {step3}")

#     step4 = step3.rearrange("b (h w)")
#     print(f"After rearrange: {step4}")

#     step5 = step4.reduce("b", "sum")
#     print(f"Final: {step5}")
