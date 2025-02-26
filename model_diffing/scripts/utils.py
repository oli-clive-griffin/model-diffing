from collections.abc import Callable, Iterator
from itertools import islice
from typing import Any

import numpy as np
import torch
import wandb
import wandb.plot.custom_chart
from einops import reduce
from schedulefree import ScheduleFreeWrapper
from torch.optim import Optimizer
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.analysis import metrics
from model_diffing.log import logger
from model_diffing.scripts.config_common import AdamConfig, BaseExperimentConfig, ScheduleFreeSigNumConfig
from model_diffing.utils import l0_norm, l2_norm


def get_l0_stats(hidden_BH: torch.Tensor) -> dict[str, float]:
    l0_BH = l0_norm(hidden_BH, dim=-1)
    mean_l0 = l0_BH.mean().item()
    l0_np = l0_BH.detach().cpu().numpy()
    l0_5, l0_25, l0_75, l0_95 = np.percentile(l0_np, [5, 25, 75, 95])
    return {
        "train/mean_firing_pct": mean_l0 / hidden_BH.shape[1],
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


def build_optimizer(
    cfg: AdamConfig | ScheduleFreeSigNumConfig, params: Iterator[torch.nn.Parameter]
) -> torch.optim.Optimizer | ScheduleFreeWrapper:
    match cfg:
        case AdamConfig():
            optimizer = torch.optim.Adam(params, lr=cfg.learning_rate, betas=cfg.betas)
        case ScheduleFreeSigNumConfig():
            optimizer = ScheduleFreeWrapper(SignSGD(params, lr=cfg.learning_rate), momentum=cfg.momentum)
            optimizer.train()
    logger.info(f"using optimizer: {optimizer}")
    return optimizer


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


@torch.no_grad()
def estimate_norm_scaling_factor_X(
    dataloader_BXD: Iterator[torch.Tensor],
    n_tokens_for_norm_estimate: int,
) -> torch.Tensor:
    batch_size, *_, d_model = next(dataloader_BXD).shape
    mean_norms_X = _estimate_mean_norms_X(dataloader_BXD, n_tokens_for_norm_estimate, batch_size)
    scaling_factors_X = torch.sqrt(torch.tensor(d_model)) / mean_norms_X
    return scaling_factors_X


@torch.no_grad()
# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norms_X(
    dataloader_BMPD: Iterator[torch.Tensor],
    n_tokens_for_norm_estimate: int,
    batch_size: int,
) -> torch.Tensor:
    norm_samples = []

    n_batches_needed = (n_tokens_for_norm_estimate + batch_size - 1) // batch_size
    for batch_BXD in tqdm(
        islice(dataloader_BMPD, n_batches_needed),
        desc="Estimating norm scaling factor",
        total=n_batches_needed,
    ):
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


class SignSGD(Optimizer):
    """Steepest descent in the L-infty norm. From <https://arxiv.org/abs/1802.04434>"""

    def __init__(self, params: Any, lr: float = 1e-3):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super(SignSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore
        assert closure is None, "Closure is not supported."

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad.sign(), alpha=-lr)
