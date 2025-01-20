from collections.abc import Iterator
from itertools import islice

import numpy as np
import torch
from einops import reduce
from tqdm import tqdm

from model_diffing.utils import l2_norm


@torch.no_grad()
def estimate_norm_scaling_factor(
    dataloader_BMLD: Iterator[torch.Tensor], d_model: int, n_batches_for_norm_estimate: int
) -> torch.Tensor:
    mean_norm = _estimate_mean_norm(dataloader_BMLD, n_batches_for_norm_estimate)
    scaling_factor = np.sqrt(d_model) / mean_norm
    return scaling_factor


@torch.no_grad()
# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norm(dataloader_BMLD: Iterator[torch.Tensor], n_batches_for_norm_estimate: int) -> float:
    norms_per_batch = []
    for batch_BMLD in tqdm(
        islice(dataloader_BMLD, n_batches_for_norm_estimate),
        desc="Estimating norm scaling factor",
    ):
        norms_BML = reduce(batch_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
        norms_mean = norms_BML.mean().item()
        print(f"- Norms mean: {norms_mean}")
        norms_per_batch.append(norms_mean)
    mean_norm = float(np.mean(norms_per_batch))
    return mean_norm
