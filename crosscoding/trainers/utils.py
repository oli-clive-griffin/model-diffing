from collections.abc import Callable, Iterator
from itertools import islice
from typing import Any, TypeVar

import numpy as np
import torch
import wandb
import wandb.plot.custom_chart
from einops import reduce
from schedulefree import ScheduleFreeWrapper  # type: ignore
from torch.optim import Optimizer
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from crosscoding.trainers.config_common import (
    AdamConfig,
    BaseExperimentConfig,
    OptimizerCfg,
    ScheduleFreeSigNumConfig,
)
from crosscoding.utils import (
    compute_cosine_similarities_N,
    compute_relative_norms_N,
    get_shared_latent_mask,
    l0_norm,
    l2_norm,
)


def get_l0_stats(hidden_BL: torch.Tensor, name: str = "l0") -> dict[str, float]:
    l0_BL = l0_norm(hidden_BL, dim=-1)
    mean_l0 = l0_BL.mean().item()
    l0_np = l0_BL.detach().cpu().numpy()
    l0_5, l0_95 = np.percentile(l0_np, [5, 95])
    return {
        f"train/{name}/mean_firing_pct": mean_l0 / hidden_BL.shape[1],
        f"train/{name}/5th": l0_5,
        f"train/{name}/mean": mean_l0,
        f"train/{name}/95th": l0_95,
    }


def create_cosine_sim_and_relative_norm_histograms(W_dec_LMD: torch.Tensor) -> tuple[wandb.Histogram, wandb.Histogram]:
    _, n_models, _ = W_dec_LMD.shape
    assert n_models == 2, "only works for 2 models"

    W_dec_m1_LD = W_dec_LMD[:, 0]
    W_dec_m2_LD = W_dec_LMD[:, 1]
    relative_norms = compute_relative_norms_N(W_dec_m1_LD, W_dec_m2_LD)
    relative_decoder_norms_plot = wandb_histogram(relative_norms)

    shared_latent_mask = get_shared_latent_mask(relative_norms)
    cosine_sims = compute_cosine_similarities_N(W_dec_m1_LD, W_dec_m2_LD)
    shared_features_cosine_sims = cosine_sims[shared_latent_mask]
    shared_features_cosine_sims_plot = wandb_histogram(shared_features_cosine_sims)

    return relative_decoder_norms_plot, shared_features_cosine_sims_plot


def create_cosine_sim_and_relative_norm_histograms_diffing(W_dec_LMD: torch.Tensor) -> dict[str, wandb.Histogram]:
    _, n_models, _ = W_dec_LMD.shape
    assert n_models == 2, "only works for 2 models"

    plots: dict[str, wandb.Histogram] = {}
    W_dec_a_LD = W_dec_LMD[:, 0]
    W_dec_b_LD = W_dec_LMD[:, 1]

    relative_norms = compute_relative_norms_N(W_dec_a_LD, W_dec_b_LD)
    plots["media/relative_decoder_norms"] = wandb_histogram(relative_norms)

    shared_latent_mask = get_shared_latent_mask(relative_norms)
    cosine_sims = compute_cosine_similarities_N(W_dec_a_LD, W_dec_b_LD)
    shared_features_cosine_sims = cosine_sims[shared_latent_mask]
    plots["media/cosine_sim"] = wandb_histogram(shared_features_cosine_sims)

    return plots


def wandb_histogram(data_X: torch.Tensor, bins: int = 100) -> wandb.Histogram:
    return wandb.Histogram(
        np_histogram=np.histogram(
            data_X.detach().cpu().numpy(),
            bins=bins,
        )
    )


def build_wandb_run(config: BaseExperimentConfig) -> Run:
    return wandb.init(
        name=config.experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=config.model_dump(),
        mode=config.wandb.mode,
    )


def build_optimizer(  # type: ignore
    cfg: OptimizerCfg, params: Iterator[torch.nn.Parameter]
) -> torch.optim.Optimizer | ScheduleFreeWrapper:
    match cfg:
        case AdamConfig():
            return torch.optim.Adam(params, lr=cfg.learning_rate, betas=cfg.betas)
        case ScheduleFreeSigNumConfig():
            optimizer = ScheduleFreeWrapper(SignSGD(params, lr=cfg.learning_rate), momentum=cfg.momentum)
            optimizer.train()
            return optimizer
    raise ValueError(f"Unknown optimizer. {cfg=}")


def build_lr_scheduler(cfg: AdamConfig, num_steps: int) -> Callable[[int], float]:
    def _lr_scheduler(step: int) -> float:
        if step < cfg.warmup_pct * num_steps:
            return cfg.learning_rate * (step / (cfg.warmup_pct * num_steps))

        pct_until_finished = 1 - (step / num_steps)
        if pct_until_finished < cfg.warmdown_pct:
            # 1 at the last step of constant learning rate period
            # 0 at the end of training
            scale = pct_until_finished / cfg.warmdown_pct
            return cfg.learning_rate * scale

        return cfg.learning_rate

    return _lr_scheduler


def estimate_norm_scaling_factor_X(
    dataloader_BXD: Iterator[torch.Tensor],
    n_tokens_for_norm_estimate: int,
) -> torch.Tensor:
    sample_BXD = next(dataloader_BXD)
    batch_size, *_, d_model = sample_BXD.shape
    mean_norms_X = _estimate_mean_norms_X(dataloader_BXD, n_tokens_for_norm_estimate, batch_size, sample_BXD.device)
    scaling_factors_X = torch.sqrt(torch.tensor(d_model, device=sample_BXD.device)) / mean_norms_X
    return scaling_factors_X


# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norms_X(
    dataloader_BMPD: Iterator[torch.Tensor],
    n_tokens_for_norm_estimate: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    norm_samples = []

    n_batches_needed = (n_tokens_for_norm_estimate + batch_size - 1) // batch_size
    for batch_BXD in tqdm(
        islice(dataloader_BMPD, n_batches_needed),
        desc="Estimating norm scaling factor",
        total=n_batches_needed,
    ):
        norms_means_X = l2_norm(batch_BXD.to(device), dim=-1).mean(dim=0)
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


class SignSGD(Optimizer):
    """Steepest descent in the L-infty norm. From <https://arxiv.org/abs/1802.04434>"""

    def __init__(self, params: Any, lr: float = 1e-3):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore
        assert closure is None, "Closure is not supported."

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad.sign(), alpha=-lr)


T = TypeVar("T")


def dict_join(dicts: list[dict[str, T]]) -> dict[str, list[T]]:
    return {k: [d[k] for d in dicts] for k in dicts[0]}
