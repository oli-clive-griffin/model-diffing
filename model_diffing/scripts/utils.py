from collections.abc import Iterator
from itertools import islice

import torch
from einops import reduce
from tqdm import tqdm

from model_diffing.utils import l2_norm, multi_reduce


@torch.no_grad()
def estimate_norm_scaling_factor_ML(
    dataloader_BMLD: Iterator[torch.Tensor],
    num_models: int,
    num_layers: int,
    d_model: int,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    mean_norms_ML = _estimate_mean_norms_ML(dataloader_BMLD, num_models, num_layers, n_batches_for_norm_estimate)
    scaling_factors_ML = torch.sqrt(torch.tensor(d_model)) / mean_norms_ML
    assert scaling_factors_ML.shape == (num_models, num_layers)
    return scaling_factors_ML


@torch.no_grad()
# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norms_ML(
    dataloader_BMLD: Iterator[torch.Tensor],
    num_models: int,
    num_layers: int,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    sample_shape_ML = (num_models, num_layers)
    norm_samples_NML = torch.empty(n_batches_for_norm_estimate, *sample_shape_ML)

    for i, batch_BMLD in tqdm(
        enumerate(islice(dataloader_BMLD, n_batches_for_norm_estimate)),
        desc="Estimating norm scaling factor",
    ):
        norms_means_ML = multi_reduce(
            batch_BMLD, "batch model layer d_model", [("d_model", l2_norm), ("batch", torch.mean)]
        )
        norm_samples_NML[i] = norms_means_ML

    mean_norms_ML = reduce(norm_samples_NML, "batch model layer -> model layer", torch.mean)
    return mean_norms_ML
