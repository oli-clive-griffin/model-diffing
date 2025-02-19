from collections.abc import Callable, Iterator
from itertools import islice
from typing import Any

import numpy as np
import torch
import wandb
import wandb.plot.custom_chart
from einops import reduce
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.analysis import metrics
from model_diffing.scripts.config_common import AdamDecayTo0LearningRateConfig, BaseExperimentConfig
from model_diffing.utils import l2_norm


def get_l0_stats(l0_B: torch.Tensor, step: int) -> dict[str, float]:
    mean_l0 = l0_B.mean().item()
    l0_np = l0_B.detach().cpu().numpy()
    l0_5, l0_25, l0_75, l0_95 = np.percentile(l0_np, [5, 25, 75, 95])
    return {
        "train/l0/step": step,
        "train/l0/5th": l0_5,
        "train/l0/25th": l0_25,
        "train/l0/mean": mean_l0,
        "train/l0/75th": l0_75,
        "train/l0/95th": l0_95,
    }


def create_cosine_sim_and_relative_norm_histograms(
    W_dec_HMPD: torch.Tensor, hookpoints: list[str]
) -> dict[str, wandb.Histogram]:
    _, n_models, num_hookpoints, _ = W_dec_HMPD.shape
    assert n_models == 2, "only works for 2 models"

    plots: dict[str, wandb.Histogram] = {}
    for hookpoint_idx in range(num_hookpoints):
        hookpoint_name = hookpoints[hookpoint_idx]
        W_dec_a_HD = W_dec_HMPD[:, 0, hookpoint_idx]
        W_dec_b_HD = W_dec_HMPD[:, 1, hookpoint_idx]

        relative_norms = metrics.compute_relative_norms_N(W_dec_a_HD, W_dec_b_HD)
        plots[f"media/relative_decoder_norms_{hookpoint_name}"] = wandb_histogram(relative_norms)

        shared_latent_mask = metrics.get_shared_latent_mask(relative_norms)
        cosine_sims = metrics.compute_cosine_similarities_N(W_dec_a_HD, W_dec_b_HD)
        shared_features_cosine_sims = cosine_sims[shared_latent_mask]
        plots[f"media/cosine_sim_{hookpoint_name}"] = wandb_histogram(shared_features_cosine_sims)

    return plots


def wandb_histogram(data_X: torch.Tensor | np.ndarray[Any, Any], bins: int = 100) -> wandb.Histogram:
    if isinstance(data_X, torch.Tensor):
        data_X = data_X.detach().cpu().numpy()
    return wandb.Histogram(np_histogram=np.histogram(data_X, bins=bins))


def build_wandb_run(config: BaseExperimentConfig) -> Run | None:
    if config.wandb == "disabled":
        return None
    return wandb.init(
        name=config.experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=config.model_dump(),
    )


def build_optimizer(cfg: AdamDecayTo0LearningRateConfig, params: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
    initial_lr = cfg.initial_learning_rate
    optimizer = torch.optim.Adam(params, lr=initial_lr)
    return optimizer


def build_lr_scheduler(cfg: AdamDecayTo0LearningRateConfig, num_steps: int) -> Callable[[int], float]:
    def _lr_scheduler(step: int) -> float:
        if step < cfg.warmup_pct * num_steps:
            return cfg.initial_learning_rate * (step / (cfg.warmup_pct * num_steps))

        pct_until_finished = 1 - (step / num_steps)
        if pct_until_finished < cfg.last_pct_of_steps:
            # 1 at the last step of constant learning rate period
            # 0 at the end of training
            scale = pct_until_finished / cfg.last_pct_of_steps
            return cfg.initial_learning_rate * scale

        return cfg.initial_learning_rate

    return _lr_scheduler


@torch.no_grad()
def estimate_norm_scaling_factor_X(
    dataloader_BXD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    d_model = next(dataloader_BXD).shape[-1]
    mean_norms_X = _estimate_mean_norms_X(dataloader_BXD, device, n_batches_for_norm_estimate)
    scaling_factors_X = torch.sqrt(torch.tensor(d_model)) / mean_norms_X
    return scaling_factors_X


@torch.no_grad()
# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norms_X(
    dataloader_BMPD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    norm_samples = []

    for batch_BXD in tqdm(
        islice(dataloader_BMPD, n_batches_for_norm_estimate),
        desc="Estimating norm scaling factor",
        total=n_batches_for_norm_estimate,
    ):
        batch_BXD = batch_BXD.to(device)
        norms_means_X = l2_norm(batch_BXD, dim=-1).mean(dim=0)
        norm_samples.append(norms_means_X)

    norm_samples_NX = torch.stack(norm_samples, dim=0)
    mean_norms_X = reduce(norm_samples_NX, "n_samples ... -> ...", torch.mean)
    return mean_norms_X


@torch.no_grad()
def collect_norms_NMP(
    dataloader_BMPD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches: int,
) -> torch.Tensor:
    norm_samples = []

    for batch_BMPD in tqdm(
        islice(dataloader_BMPD, n_batches),
        desc="Collecting norms",
        total=n_batches,
    ):
        batch_BMPD = batch_BMPD.to(device)
        norms_BMP = reduce(batch_BMPD, "batch model hookpoint d_model -> batch model hookpoint", l2_norm)
        norm_samples.append(norms_BMP)

    norm_samples_NMP = torch.cat(norm_samples, dim=0)
    return norm_samples_NMP
