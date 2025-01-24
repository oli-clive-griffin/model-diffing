from abc import abstractmethod
from collections.abc import Iterator

import torch
import wandb
from einops import einsum
from wandb.sdk.wandb_run import Run

from model_diffing.analysis.visualization import create_visualizations
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer, estimate_norm_scaling_factor_ML
from model_diffing.utils import save_model_and_config


class BaseTrainer[TConfig: BaseTrainConfig]:
    def __init__(
        self,
        cfg: TConfig,
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
        self.dataloader_BMLD = dataloader_BMLD
        self.wandb_run = wandb_run
        self.device = device
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

            if self.wandb_run:
                if (self.step + 1) % self.cfg.log_every_n_steps == 0:
                    self.wandb_run.log(log_dict, step=self.step)

                if (self.step + 1) % self.cfg.log_visualizations_every_n_steps == 0:
                    visualizations = create_visualizations(
                        self.crosscoder.W_dec_HMLD.detach().cpu(), self.layers_to_harvest
                    )
                    self.wandb_run.log(
                        {f"visualizations/{k}": wandb.Plotly(v) for k, v in visualizations.items()},
                        step=self.step,
                    )

            if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
                save_model_and_config(
                    config=self.cfg,
                    save_dir=self.cfg.save_dir,
                    model=self.crosscoder,
                    epoch=self.step,
                )

            self.step += 1

    def _next_batch_BMLD(self, norm_scaling_factors_ML: torch.Tensor) -> torch.Tensor:
        batch_BMLD = next(self.dataloader_BMLD)
        batch_BMLD = batch_BMLD.to(self.device)
        batch_BMLD = einsum(
            batch_BMLD, norm_scaling_factors_ML, "batch model layer d_model, model layer -> batch model layer d_model"
        )
        return batch_BMLD

    @abstractmethod
    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]: ...
