from typing import Any

import torch

from crosscoding.models.activations.relu import ReLUActivation
from crosscoding.models.sparse_coders import CrossLayerTranscoder
from crosscoding.trainers.train_l1_crosscoder.trainer import sparsity_loss_l1_of_l2s
from crosscoding.trainers.train_l1_crosslayer_trancoder.base_transcoder_trainer import BaseCrossLayerTranscoderTrainer
from crosscoding.trainers.train_l1_crosslayer_trancoder.config import L1TrainConfig
from crosscoding.trainers.utils import get_l0_stats
from crosscoding.utils import calculate_reconstruction_loss_summed_norm_MSEs


class L1CrossLayerTranscoderTrainer(BaseCrossLayerTranscoderTrainer[L1TrainConfig, ReLUActivation]):
    def _calculate_loss_and_log(
        self,
        train_res: CrossLayerTranscoder.ForwardResult,
        target_BPD: torch.Tensor,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(train_res.output_BPD, target_BPD)

        sparsity_loss = sparsity_loss_l1_of_l2s(self.crosscoder.W_dec_LPD, train_res.latents_BL)

        loss = reconstruction_loss + self._l1_coef_scheduler() * sparsity_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(target_BPD, train_res.output_BPD),
                **get_l0_stats(train_res.latents_BL),
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
            