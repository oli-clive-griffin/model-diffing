from typing import Any

import torch

from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.topk import TopkActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict


class TopKTrainer(BaseModelHookpointTrainer[BaseTrainConfig, TopkActivation]):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        logger.warn("Auxiliary loss is not implemented for topk training, you may see large amounts of dead latents")

    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
    ) -> torch.Tensor:
        loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.recon_acts_BXD)

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
                "train/loss": loss.item(),
                **fvu_dict,
                **self._common_logs(),
            }

            self.wandb_run.log(log_dict, step=self.step)

        return loss
