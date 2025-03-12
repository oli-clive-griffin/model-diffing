from typing import Any

import torch as t

from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.train_jan_update_crosscoder.config import TanHSparsityTrainConfig
from model_diffing.scripts.utils import get_l0_stats, wandb_histogram
from model_diffing.utils import (
    calculate_reconstruction_loss_summed_MSEs,
    get_summed_decoder_norms_H,
    pre_act_loss,
    tanh_sparsity_loss,
)


class JanUpdateCrosscoderTrainer(BaseModelHookpointTrainer[TanHSparsityTrainConfig, AnthropicJumpReLUActivation]):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: t.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[t.Tensor, dict[str, Any] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.recon_acts_BXD)
        decoder_norms_H = get_summed_decoder_norms_H(self.crosscoder.W_dec_HXD)
        tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.hidden_BH, decoder_norms_H)
        pre_act_loss = self._pre_act_loss(train_res.hidden_BH, decoder_norms_H)

        loss = reconstruction_loss + self._lambda_s_scheduler() * tanh_sparsity_loss + self.cfg.lambda_p * pre_act_loss

        if log:
            log_dict: dict[str, float] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/pre_act_loss": pre_act_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(batch_BMPD, train_res.recon_acts_BXD),
                **get_l0_stats(train_res.hidden_BH),
            }

            return loss, log_dict

        return loss, None

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return tanh_sparsity_loss(self.cfg.c, hidden_BH, decoder_norms_H)

    def _pre_act_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return pre_act_loss(self.crosscoder.hidden_activation.log_threshold_H, hidden_BH, decoder_norms_H)

    def _step_logs(self) -> dict[str, Any]:
        log_dict = {
            **super()._step_logs(),
            "train/lambda_s": self._lambda_s_scheduler(),
            "train/lambda_p": self.cfg.lambda_p,
        }
        if self.cfg.log_every_n_steps is not None and self.step % (self.cfg.log_every_n_steps * 10) == 0:
            threshold_hist = wandb_histogram(self.crosscoder.hidden_activation.log_threshold_H.exp())
            log_dict.update({"media/jumprelu_threshold_distribution": threshold_hist})
        return log_dict
