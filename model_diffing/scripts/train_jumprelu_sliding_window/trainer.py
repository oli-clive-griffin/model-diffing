from typing import Any

import torch as t

from model_diffing.models.activations.jumprelu import AnthropicSTEJumpReLUActivation
from model_diffing.scripts.base_sliding_window_trainer import BaseSlidingWindowCrosscoderTrainer, BiTokenCCWrapper
from model_diffing.scripts.train_jan_update_crosscoder.config import TanHSparsityTrainConfig
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import calculate_reconstruction_loss_summed_norm_MSEs, get_summed_decoder_norms_L


class JumpReLUSlidingWindowCrosscoderTrainer(
    BaseSlidingWindowCrosscoderTrainer[AnthropicSTEJumpReLUActivation, TanHSparsityTrainConfig]
):
    def _calculate_loss_and_log(
        self,
        batch_BTPD: t.Tensor,
        res: BiTokenCCWrapper.TrainResult,
        log: bool,
    ) -> tuple[t.Tensor, dict[str, float] | None]:
        reconstructed_acts_BTPD = t.cat([res.recon_single1_B1PD, res.recon_single2_B1PD], dim=1) + res.recon_double_B2PD
        assert reconstructed_acts_BTPD.shape == batch_BTPD.shape, "fuck"

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(batch_BTPD, reconstructed_acts_BTPD)

        decoder_norms_single_L = get_summed_decoder_norms_L(self.crosscoders.single_cc.W_dec_LXD)
        decoder_norms_both_L = get_summed_decoder_norms_L(self.crosscoders.double_cc.W_dec_LXD)

        decoder_norms_3l = t.cat([decoder_norms_single_L, decoder_norms_both_L, decoder_norms_single_L], dim=-1)
        hidden_B3l = t.cat([res.hidden_single1_BL, res.hidden_double_BL, res.hidden_single2_BL], dim=-1)

        tanh_sparsity_loss = self._tanh_sparsity_loss(hidden_B3l, decoder_norms_3l)
        lambda_s = self._lambda_s_scheduler()

        pre_act_loss = self._pre_act_loss(hidden_B3l, decoder_norms_3l)

        loss = (
            reconstruction_loss  #
            + lambda_s * tanh_sparsity_loss
            + self.cfg.lambda_p * pre_act_loss
        )

        if log:
            log_dict: dict[str, float] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/pre_act_loss": pre_act_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(batch_BTPD, reconstructed_acts_BTPD),
                **get_l0_stats(hidden_B3l),
            }

            return loss, log_dict

        return loss, None

    def _step_logs(self) -> dict[str, Any]:
        log_dict: dict[str, Any] = {
            **super()._step_logs(),
            "train/lambda_s": self._lambda_s_scheduler(),
            "train/lambda_p": self.cfg.lambda_p,
        }
        return log_dict

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BL: t.Tensor, decoder_norms_L: t.Tensor) -> t.Tensor:
        loss_BL = t.tanh(self.cfg.c * hidden_BL * decoder_norms_L)
        return loss_BL.sum(-1).mean()

    def _pre_act_loss(self, latent_B3l: t.Tensor, decoder_norms_3l: t.Tensor) -> t.Tensor:
        t_3l = t.cat(
            [
                self.crosscoders.single_cc.activation_fn.log_threshold_L,
                self.crosscoders.double_cc.activation_fn.log_threshold_L,
                self.crosscoders.single_cc.activation_fn.log_threshold_L,
            ],
            dim=-1,
        )
        loss_B3l = t.relu(t_3l.exp() - latent_B3l) * decoder_norms_3l
        return loss_B3l.sum(-1).mean()
