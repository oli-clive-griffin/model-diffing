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
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.recon_acts_BXD)

        if log:
            fvu_dict = get_fvu_dict(
                batch_BMPD,
                train_res.recon_acts_BXD,
                ("model", list(range(self.n_models))),
                ("hookpoint", self.hookpoints),
            )

            log_dict: dict[str, Any] = {
                "train/loss": loss.item(),
                **fvu_dict,
            }

            return loss, log_dict

        return loss, None
