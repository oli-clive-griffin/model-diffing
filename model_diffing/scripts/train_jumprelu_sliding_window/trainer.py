from typing import Any

import torch as t
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.train_jan_update_crosscoder.config import TanHSparsityTrainConfig
from model_diffing.scripts.base_sliding_window_trainer import BaseSlidingWindowCrosscoderTrainer
from model_diffing.scripts.utils import get_l0_stats, wandb_histogram
from model_diffing.utils import (
    calculate_reconstruction_loss_summed_MSEs,
    get_fvu_dict,
    get_summed_decoder_norms_H,
)


class JumpReLUSlidingWindowCrosscoderTrainer(
    BaseSlidingWindowCrosscoderTrainer[AnthropicJumpReLUActivation, TanHSparsityTrainConfig]
):
    def _train_step(self, batch_BTPD: t.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        # fwd
        res = self.crosscoders.forward_train(batch_BTPD)

        reconstructed_acts_BTPD = t.cat([res.recon_B1PD_single1, res.recon_B1PD_single2], dim=1) + res.recon_B2PD_double
        assert reconstructed_acts_BTPD.shape == batch_BTPD.shape, "fuck"

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BTPD, reconstructed_acts_BTPD)

        hidden_B3h = t.cat([res.hidden_BH_single1, res.hidden_BH_double, res.hidden_BH_single2], dim=-1)

        decoder_norms_single_H = get_summed_decoder_norms_H(self.crosscoders.single_cc.W_dec_HXD)
        decoder_norms_both_H = get_summed_decoder_norms_H(self.crosscoders.double_cc.W_dec_HXD)

        decoder_norms_3h = t.cat([decoder_norms_single_H, decoder_norms_both_H, decoder_norms_single_H], dim=-1)

        tanh_sparsity_loss = self._tanh_sparsity_loss(hidden_B3h, decoder_norms_3h)
        pre_act_loss = self._pre_act_loss(hidden_B3h, decoder_norms_3h)

        lambda_s = self._lambda_s_scheduler()
        scaled_tanh_sparsity_loss = lambda_s * tanh_sparsity_loss
        scaled_pre_act_loss = self.cfg.lambda_p * pre_act_loss

        loss = (
            reconstruction_loss  #
            + scaled_tanh_sparsity_loss
            + scaled_pre_act_loss
        )

        # backward
        loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()

        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            clip_grad_norm_(self.crosscoders.parameters(), 1.0)
            self.optimizer.step()

        self._lr_step()

        self.firing_tracker.add_batch(hidden_B3h)

        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and self.step % self.cfg.log_every_n_steps == 0
        ):
            fvu_dict = get_fvu_dict(
                batch_BTPD,
                reconstructed_acts_BTPD,
                ("token", [0, 1]),
                ("hookpoint", self.hookpoints),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/tanh_sparsity_loss_scaled": scaled_tanh_sparsity_loss.item(),
                "train/lambda_s": lambda_s,
                "train/pre_act_loss": pre_act_loss.item(),
                "train/pre_act_loss_scaled": scaled_pre_act_loss.item(),
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

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

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
