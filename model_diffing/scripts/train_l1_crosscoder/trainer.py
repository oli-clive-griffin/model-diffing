from collections.abc import Iterator
from dataclasses import dataclass

import torch
import wandb
from einops import einsum
from torch.nn.utils import clip_grad_norm_
from wandb.sdk.wandb_run import Run

from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_l1_crosscoder.config import TrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer, estimate_norm_scaling_factor_ML
from model_diffing.utils import (
    calculate_explained_variance_ML,
    calculate_reconstruction_loss,
    get_explained_var_dict,
    l0_norm,
    multi_reduce,
    save_model_and_config,
    sparsity_loss_l1_of_norms,
)


@dataclass
class LossInfo:
    l1_coef: float
    reconstruction_loss: float
    sparsity_loss: float
    mean_l0: float
    explained_variance_ML: torch.Tensor


class L1CrosscoderTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataloader_BMLD: Iterator[torch.Tensor],
        crosscoder: AcausalCrosscoder,
        wandb_run: Run | None,
        device: torch.device,
        layers_to_harvest: list[int],
    ):
        self.cfg = cfg
        self.crosscoder = crosscoder
        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())
        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, cfg.num_steps)
        self.wandb_run = wandb_run
        self.device = device
        self.dataloader_BMLD = dataloader_BMLD
        self.layers_to_harvest = layers_to_harvest

        self.step = 0
        self.tokens_trained = 0

    def train(self):
        logger.info("Estimating norm scaling factors (model, layer)")

        norm_scaling_factors_ML = estimate_norm_scaling_factor_ML(
            self.dataloader_BMLD,
            self.device,
            self.cfg.n_batches_for_norm_estimate,
        )

        logger.info(f"Norm scaling factors (model, layer): {norm_scaling_factors_ML}")

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )

        while self.step < self.cfg.num_steps:
            batch_BMLD = self._next_batch_BMLD(norm_scaling_factors_ML)

            log_dict = self._train_step(batch_BMLD)

            if self.wandb_run and (self.step + 1) % self.cfg.log_every_n_steps == 0:
                self.wandb_run.log(log_dict, step=self.step)

            if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
                save_model_and_config(
                    config=self.cfg,
                    save_dir=self.cfg.save_dir,
                    model=self.crosscoder,
                    epoch=self.step,
                )

            self.step += 1

    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMLD)
        self.tokens_trained += batch_BMLD.shape[0]

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BMLD)
        sparsity_loss = sparsity_loss_l1_of_norms(self.crosscoder.W_dec_HMLD, train_res.hidden_BH)
        l1_coef = self._l1_coef_scheduler()
        loss = reconstruction_loss + l1_coef * sparsity_loss

        # backward
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        # metrics
        mean_l0 = multi_reduce(train_res.hidden_BH, "batch hidden", ("hidden", l0_norm), ("batch", torch.mean)).item()
        explained_variance_ML = calculate_explained_variance_ML(batch_BMLD, train_res.reconstructed_acts_BMLD)

        log_dict = {
            "train/l1_coef": l1_coef,
            "train/mean_l0": mean_l0,
            "train/mean_l0_pct": mean_l0 / self.crosscoder.hidden_dim,
            "train/reconstruction_loss": reconstruction_loss.item(),
            "train/sparsity_loss": sparsity_loss.item(),
            "train/loss": loss.item(),
            "train/tokens_trained": self.tokens_trained,
            **get_explained_var_dict(explained_variance_ML, self.layers_to_harvest),
        }

        return log_dict

    def _next_batch_BMLD(self, norm_scaling_factors_ML: torch.Tensor) -> torch.Tensor:
        batch_BMLD = next(self.dataloader_BMLD)
        batch_BMLD = batch_BMLD.to(self.device)
        batch_BMLD = einsum(
            batch_BMLD, norm_scaling_factors_ML, "batch model layer d_model, model layer -> batch model layer d_model"
        )
        return batch_BMLD

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.l1_coef_n_steps:
            return self.cfg.l1_coef_max * self.step / self.cfg.l1_coef_n_steps
        else:
            return self.cfg.l1_coef_max
