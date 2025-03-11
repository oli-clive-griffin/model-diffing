from typing import Any

import torch as t

from model_diffing.models.acausal_crosscoder.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.train_jan_update_crosscoder.config import TanHSparsityTrainConfig
from model_diffing.scripts.utils import get_l0_stats, wandb_histogram
from model_diffing.utils import (
    calculate_reconstruction_loss_summed_MSEs,
    get_fvu_dict,
    get_summed_decoder_norms_H,
    pre_act_loss,
    tanh_sparsity_loss,
)


class JanUpdateCrosscoderTrainer(BaseModelHookpointTrainer[TanHSparsityTrainConfig, AnthropicJumpReLUActivation]):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: t.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
    ) -> t.Tensor:
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.recon_acts_BXD)

        decoder_norms_H = get_summed_decoder_norms_H(self.crosscoder.W_dec_HXD)

        tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.hidden_BH, decoder_norms_H)
        lambda_s = self._lambda_s_scheduler()

        pre_act_loss = self._pre_act_loss(train_res.hidden_BH, decoder_norms_H)

        loss = (
            reconstruction_loss  #
            + lambda_s * tanh_sparsity_loss
            + self.cfg.lambda_p * pre_act_loss
        )

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BMPD,
                train_res.recon_acts_BXD,
                ("model", list(range(self.n_models))),
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
                **get_l0_stats(train_res.hidden_BH),
                **self._common_logs(),
            }

            if self.step % (self.cfg.log_every_n_steps * 10) == 0:
                log_dict.update(
                    {
                        "media/jumprelu_threshold_distribution": wandb_histogram(
                            self.crosscoder.hidden_activation.log_threshold_H.exp()
                        ),
                    }
                )

            self.wandb_run.log(log_dict, step=self.step)

        return loss

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return tanh_sparsity_loss(self.cfg.c, hidden_BH, decoder_norms_H)

    def _pre_act_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return pre_act_loss(self.crosscoder.hidden_activation.log_threshold_H, hidden_BH, decoder_norms_H)
