from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from model_diffing.log import logger
from model_diffing.models.activations.topk import TopkActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict


class TopKTrainer(BaseModelHookpointTrainer[BaseTrainConfig, TopkActivation]):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        logger.warn("Auxiliary loss is not implemented for topk training, you may see large amounts of dead latents")

    def _train_step(self, batch_BMPD: torch.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMPD)
        self.firing_tracker.add_batch(train_res.hidden_BH)

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.recon_acts_BXD)

        # backward
        reconstruction_loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()

        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()

        self._lr_step()

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BMPD,
                train_res.recon_acts_BXD,
                ("model", list(range(self.n_models))),
                ("hookpoint", self.hookpoints),
            )

            log_dict: dict[str, Any] = {
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/reconstruction_loss": reconstruction_loss.item(),
                **fvu_dict,
                **self._common_logs(),
            }

            self.wandb_run.log(log_dict, step=self.step)
