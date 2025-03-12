from typing import Any

import torch

from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, sparsity_loss_l1_of_norms


class L1CrosscoderTrainer(BaseModelHookpointTrainer[L1TrainConfig, ReLUActivation]):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.recon_acts_BXD)

        sparsity_loss = sparsity_loss_l1_of_norms(
            W_dec_HTMPD=self.crosscoder.W_dec_HXD[:, None],
            hidden_BH=train_res.hidden_BH,
        )

        loss = reconstruction_loss + self._l1_coef_scheduler() * sparsity_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(batch_BMPD, train_res.recon_acts_BXD),
                **get_l0_stats(train_res.hidden_BH),
            }

            return loss, log_dict

        return loss, None
    
    def _step_logs(self) -> dict[str, Any]:
        return {
            **super()._step_logs(),
            "train/l1_coef": self._l1_coef_scheduler(),
        }

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.lambda_s_n_steps:
            return self.cfg.final_lambda_s * self.step / self.cfg.lambda_s_n_steps
        else:
            return self.cfg.final_lambda_s
