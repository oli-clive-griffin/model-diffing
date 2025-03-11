from typing import Any

import torch as t

from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.base_sliding_window_trainer import BaseSlidingWindowCrosscoderTrainer, BiTokenCCWrapper
from model_diffing.scripts.train_jan_update_crosscoder.config import TanHSparsityTrainConfig
from model_diffing.scripts.utils import get_l0_stats, wandb_histogram
from model_diffing.utils import (
    calculate_reconstruction_loss_summed_MSEs,
    get_fvu_dict,
    get_summed_decoder_norms_H,
)


class JumpReLUSlidingWindowCrosscoderTrainer(
    BaseSlidingWindowCrosscoderTrainer[AnthropicJumpReLUActivation, TanHSparsityTrainConfig]
):
    def _calculate_loss_and_log(
        self,
        batch_BTPD: t.Tensor,
        res: BiTokenCCWrapper.TrainResult,
    ) -> t.Tensor:
        reconstructed_acts_BTPD = t.cat([res.recon_single1_B1PD, res.recon_single2_B1PD], dim=1) + res.recon_double_B2PD
        assert reconstructed_acts_BTPD.shape == batch_BTPD.shape, "fuck"

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BTPD, reconstructed_acts_BTPD)

        decoder_norms_single_H = get_summed_decoder_norms_H(self.crosscoders.single_cc.W_dec_HXD)
        decoder_norms_both_H = get_summed_decoder_norms_H(self.crosscoders.double_cc.W_dec_HXD)

        decoder_norms_3h = t.cat([decoder_norms_single_H, decoder_norms_both_H, decoder_norms_single_H], dim=-1)
        hidden_B3h = t.cat([res.hidden_single1_BH, res.hidden_double_BH, res.hidden_single2_BH], dim=-1)

        tanh_sparsity_loss = self._tanh_sparsity_loss(hidden_B3h, decoder_norms_3h)
        lambda_s = self._lambda_s_scheduler()

        pre_act_loss = self._pre_act_loss(hidden_B3h, decoder_norms_3h)

        loss = (
            reconstruction_loss  #
            + lambda_s * tanh_sparsity_loss
            + self.cfg.lambda_p * pre_act_loss
        )

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BTPD,
                reconstructed_acts_BTPD,
                ("token", [0, 1]),
                ("hookpoint", self.hookpoints),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/lambda_s": lambda_s,
                "train/pre_act_loss": pre_act_loss.item(),
                "train/lambda_p": self.cfg.lambda_p,
                "train/loss": loss.item(),
                **fvu_dict,
                **get_l0_stats(hidden_B3h),
                **self._common_logs(),
            }

            if self.step % (self.cfg.log_every_n_steps * 10) == 0:
                log_dict.update(
                    {
                        "media/tokens_since_fired": wandb_histogram(self.firing_tracker.examples_since_fired_A),
                    }
                )

            self.wandb_run.log(log_dict, step=self.step)
        return loss

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        loss_BH = t.tanh(self.cfg.c * hidden_BH * decoder_norms_H)
        return loss_BH.sum(-1).mean()

    def _pre_act_loss(self, hidden_B3h: t.Tensor, decoder_norms_3h: t.Tensor) -> t.Tensor:
        t_3h = t.cat(
            [
                self.crosscoders.single_cc.hidden_activation.log_threshold_H,
                self.crosscoders.double_cc.hidden_activation.log_threshold_H,
                self.crosscoders.single_cc.hidden_activation.log_threshold_H,
            ],
            dim=-1,
        )
        loss_3h = t.relu(t_3h.exp() - hidden_B3h) * decoder_norms_3h
        return loss_3h.sum(-1).mean()
