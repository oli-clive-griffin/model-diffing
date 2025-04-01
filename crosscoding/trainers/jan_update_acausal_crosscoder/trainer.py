from typing import Any

import torch

from crosscoding.models.activations.jumprelu import AnthropicSTEJumpReLUActivation
from crosscoding.models.sparse_coders import ModelHookpointAcausalCrosscoder
from crosscoding.trainers.base_acausal_trainer import BaseModelHookpointAcausalTrainer
from crosscoding.trainers.jan_update_acausal_crosscoder.config import TanHSparsityTrainConfig
from crosscoding.trainers.utils import get_l0_stats, wandb_histogram
from crosscoding.utils import (
    calculate_reconstruction_loss_summed_norm_MSEs,
    get_summed_decoder_norms_L,
    not_none,
    pre_act_loss,
    tanh_sparsity_loss,
)


class JanUpdateModelHookpointAcausalCrosscoderTrainer(
    BaseModelHookpointAcausalTrainer[TanHSparsityTrainConfig, AnthropicSTEJumpReLUActivation]
):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(batch_BMPD, train_res.recon_acts_BMPD)
        decoder_norms_L = get_summed_decoder_norms_L(self.crosscoder.W_dec_LMPD)
        tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.latents_BL, decoder_norms_L)
        pre_act_loss = self._pre_act_loss(train_res.latents_BL, decoder_norms_L)

        loss = (
            reconstruction_loss  #
            + self._lambda_s_scheduler() * tanh_sparsity_loss
            + self.cfg.lambda_p * pre_act_loss
        )

        if log:
            log_dict: dict[str, float] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/pre_act_loss": pre_act_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(batch_BMPD, train_res.recon_acts_BMPD),
                **get_l0_stats(train_res.latents_BL),
            }

            return loss, log_dict

        return loss, None

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.cfg.num_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BL: torch.Tensor, decoder_norms_L: torch.Tensor) -> torch.Tensor:
        return tanh_sparsity_loss(self.cfg.c, hidden_BL, decoder_norms_L)

    def _pre_act_loss(self, hidden_BL: torch.Tensor, decoder_norms_L: torch.Tensor) -> torch.Tensor:
        return pre_act_loss(self.crosscoder.activation_fn.log_threshold_L, hidden_BL, decoder_norms_L)

    def _step_logs(self) -> dict[str, Any]:
        log_dict = super()._step_logs()
        log_dict.update(
            {
                "train/lambda_s": self._lambda_s_scheduler(),
                "train/lambda_p": self.cfg.lambda_p,
            }
        )
        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            threshold_hist = wandb_histogram(self.crosscoder.activation_fn.log_threshold_L.exp())
            log_dict.update(
                {
                    "media/jr_threshold": threshold_hist,
                    "media/jr_threshold_grad": wandb_histogram(
                        not_none(self.crosscoder.activation_fn.log_threshold_L.grad)
                    ),
                }
            )
            if self.crosscoder.b_enc_L is not None:
                log_dict["b_enc_values"] = wandb_histogram(self.crosscoder.b_enc_L)

        return log_dict
