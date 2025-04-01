import os
from collections import deque
from collections.abc import Iterator
from itertools import product
from typing import TypeVar

import einops
import torch
from pydantic import BaseModel as _BaseModel

from crosscoding.log import logger


class BaseModel(_BaseModel):
    class Config:
        extra = "forbid"


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


def calculate_reconstruction_loss_summed_norm_MSEs(
    activation_BXD: torch.Tensor,
    target_BXD: torch.Tensor,
) -> torch.Tensor:
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

    variance_of_error_X = _norm_variance(y_BXD, y_pred_BXD)

    # think of this as "the average example"
    y_mean_1XD = y_BXD.mean(dim=0, keepdim=True)
    variance_of_data_X = _norm_variance(y_BXD, y_mean_1XD)

    return variance_of_error_X / (variance_of_data_X + eps)


def _norm_variance(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    # The mean squared l2 norm of the error
    variance_of_error_BX = (y - y_hat).norm(p=2, dim=-1).square()  # variance
    variance_of_error_X = variance_of_error_BX.mean(0)
    return variance_of_error_X


def get_fvu_dict(
    y_BXD: torch.Tensor,
    y_pred_BXD: torch.Tensor,
    *crosscoding_dims: tuple[str, list[str]],
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


def get_summed_decoder_norms_L(W_dec_LXoDo: torch.Tensor) -> torch.Tensor:
    W_dec_l2_norms_LXo = einops.reduce(W_dec_LXoDo, "latent ... dim -> latent ...", l2_norm)
    norms_L = einops.reduce(W_dec_l2_norms_LXo, "latent ... -> latent", torch.sum)
    return norms_L


# useful for debugging
def inspect(tensor: torch.Tensor) -> str:
    return f"{tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={_size_human_readable(tensor)}"


def _size_human_readable(tensor: torch.Tensor) -> str:
    # Calculate the number of bytes in the tensor
    if tensor.nbytes >= 1024**3:
        return f"{tensor.nbytes / (1024**3):.2f} GB"
    elif tensor.nbytes >= 1024**2:
        return f"{tensor.nbytes / (1024**2):.2f} MB"
    elif tensor.nbytes >= 1024:
        return f"{tensor.nbytes / 1024:.2f} KB"
    else:
        return f"{tensor.nbytes} B"


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
    queue = deque[torch.Tensor]()
    cum_batch_size = 0

    for tensor_HX in iterator_HX:
        # consume as much as possible from the current tensor, potentially yielding multiple batches
        if cum_batch_size + tensor_HX.shape[0] >= new_batch_size_B:
            needed = new_batch_size_B - cum_batch_size
            taken_HX = tensor_HX[:needed]
            yield torch.cat([*queue, taken_HX])
            queue.clear()
            cum_batch_size = 0

            tensor_HX = tensor_HX[needed:]
            while tensor_HX.shape[0] > new_batch_size_B:
                yield tensor_HX[:new_batch_size_B]
                tensor_HX = tensor_HX[new_batch_size_B:]

        if tensor_HX.shape[0] > 0:
            cum_batch_size += tensor_HX.shape[0]
            queue.append(tensor_HX)

    if queue and yield_final_batch:
        yield torch.cat(list(queue))


@torch.no_grad()
def random_direction_init_(tensor: torch.Tensor, norm: float) -> None:
    tensor.normal_()
    tensor.div_(l2_norm(tensor, dim=-1, keepdim=True))
    tensor.mul_(norm)


T = TypeVar("T")


def runtimecast(thing: T, cls: type[T]) -> T:
    if not isinstance(thing, cls):
        raise ValueError(f"Expected a {cls.__name__}, got {type(thing)}")
    return thing


# hacky but useful for debugging
torch.Tensor.i = lambda self: inspect(self)  # type: ignore


def not_none(x: T | None) -> T:
    if x is None:
        raise ValueError("x is None")
    return x


def pre_act_loss(
    log_threshold_L: torch.Tensor, latents_BL: torch.Tensor, decoder_norms_L: torch.Tensor
) -> torch.Tensor:
    loss_BL = torch.relu(log_threshold_L.exp() - latents_BL) * decoder_norms_L
    return loss_BL.sum(-1).mean()


def tanh_sparsity_loss(c: float, latents_BL: torch.Tensor, decoder_norms_L: torch.Tensor) -> torch.Tensor:
    loss_BL = torch.tanh(c * latents_BL * decoder_norms_L)
    return loss_BL.sum(-1).mean()


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# N = num_vectors
# F = feature dim
def compute_relative_norms_N(vectors_a_NF: torch.Tensor, vectors_b_NF: torch.Tensor) -> torch.Tensor:
    """Compute relative norms between two sets of vectors."""
    norm_a_N = torch.norm(vectors_a_NF, dim=-1)
    norm_b_N = torch.norm(vectors_b_NF, dim=-1)
    return (norm_b_N + 1e-6) / (norm_a_N + norm_b_N + 1e-6)


def compute_cosine_similarities_N(vectors_a_NF: torch.Tensor, vectors_b_NF: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarities between corresponding vectors."""
    return torch.nn.functional.cosine_similarity(vectors_a_NF, vectors_b_NF, dim=-1)


def get_shared_latent_mask(
    relative_norms: torch.Tensor, min_thresh: float = 0.3, max_thresh: float = 0.7
) -> torch.Tensor:
    """Create mask for shared latents based on relative norms."""
    return torch.logical_and(relative_norms > min_thresh, relative_norms < max_thresh)
