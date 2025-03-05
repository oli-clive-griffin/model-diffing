from typing import Any

import torch as t
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.train_jan_update_crosscoder.config import TanHSparsityTrainConfig
from model_diffing.scripts.utils import get_l0_stats, wandb_histogram
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict, get_summed_decoder_norms_H


class JanUpdateCrosscoderTrainer(BaseModelHookpointTrainer[TanHSparsityTrainConfig, AnthropicJumpReLUActivation]):
    def _train_step(self, batch_BMPD: t.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMPD)
        self.firing_tracker.add_batch(train_res.hidden_BH)

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.output_BXD)

        decoder_norms_H = get_summed_decoder_norms_H(self.crosscoder.W_dec_HXD)
        tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.hidden_BH, decoder_norms_H)
        pre_act_loss = self._pre_act_loss(train_res.hidden_BH, decoder_norms_H)

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
            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()
            self._lr_step()

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BMPD,
                train_res.output_BXD,
                ("model", list(range(self.n_models))),
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
                **get_l0_stats(train_res.hidden_BH),
                **self._common_logs(),
            }

            if self.step % (self.cfg.log_every_n_steps * 10) == 0:
                try:
                    log_dict.update(
                        {
                            "media/jumprelu_threshold_distribution": wandb_histogram(
                                self.crosscoder.hidden_activation.log_threshold_H.exp()
                            ),
                        }
                    )
                except Exception:
                    ...
                    # logger.error(f"Error logging jumprelu_threshold_distribution: {e}")
            self.wandb_run.log(log_dict, step=self.step)

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return tanh_sparsity_loss(self.cfg.c, hidden_BH, decoder_norms_H)

    def _pre_act_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return pre_act_loss(self.crosscoder.hidden_activation.log_threshold_H, hidden_BH, decoder_norms_H)


def pre_act_loss(log_threshold_H: t.Tensor, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
    loss_BH = t.relu(log_threshold_H.exp() - hidden_BH) * decoder_norms_H
    return loss_BH.sum(-1).mean()


def tanh_sparsity_loss(c: float, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
    loss_BH = t.tanh(c * hidden_BH * decoder_norms_H)
    return loss_BH.sum(-1).mean()
