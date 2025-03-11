from typing import Any

import torch as t

from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.base_diffing_trainer import BaseDiffingTrainer
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateTrainConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import pre_act_loss, tanh_sparsity_loss
from model_diffing.scripts.utils import get_l0_stats, wandb_histogram
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict, get_summed_decoder_norms_H


class ModelDiffingFebUpdateJumpReLUTrainer(
    BaseDiffingTrainer[JumpReLUModelDiffingFebUpdateTrainConfig, AnthropicJumpReLUActivation]
):
    """
    Implementation of https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html but with jumprelu
    loss as described in https://transformer-circuits.pub/2025/january-update/index.html. Expected to be used with
    a JumpReLU crosscoder.
    """

    def _loss_and_log_dict(
        self,
        batch_BMD: t.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
    ) -> tuple[t.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMD, train_res.recon_acts_BXD)

        decoder_norms_H = get_summed_decoder_norms_H(self.crosscoder.W_dec_HXD)
        decoder_norms_shared_Hs = decoder_norms_H[: self.n_shared_latents]
        decoder_norms_indep_Hi = decoder_norms_H[self.n_shared_latents :]

        hidden_shared_BHs = train_res.hidden_BH[:, : self.n_shared_latents]
        hidden_indep_BHi = train_res.hidden_BH[:, self.n_shared_latents :]

        # shared features sparsity loss
        tanh_sparsity_loss_shared = self._tanh_sparsity_loss(hidden_shared_BHs, decoder_norms_shared_Hs)
        lambda_s = self._lambda_s_scheduler()

        # independent features sparsity loss
        tanh_sparsity_loss_indep = self._tanh_sparsity_loss(hidden_indep_BHi, decoder_norms_indep_Hi)
        lambda_f = self._lambda_f_scheduler()

        # pre-activation loss on all features
        pre_act_loss = self._pre_act_loss(train_res.hidden_BH, decoder_norms_H)

        # total loss
        loss = (
            reconstruction_loss  #
            + lambda_s * tanh_sparsity_loss_shared
            + lambda_f * tanh_sparsity_loss_indep
            + self.cfg.lambda_p * pre_act_loss
        )

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BMD,
                train_res.recon_acts_BXD,
                ("model", [0, 1]),
            )

            hidden_shared_BHs = train_res.hidden_BH[:, : self.n_shared_latents]
            hidden_indep_BHi = train_res.hidden_BH[:, self.n_shared_latents :]

            log_dict: dict[str, float] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss_shared": tanh_sparsity_loss_shared.item(),
                "train/tanh_sparsity_loss_indep": tanh_sparsity_loss_indep.item(),
                "train/pre_act_loss": pre_act_loss.item(),
                "train/loss": loss.item(),
                **fvu_dict,
                **get_l0_stats(hidden_shared_BHs, name="shared_l0"),
                **get_l0_stats(hidden_indep_BHi, name="indep_l0"),
                **get_l0_stats(train_res.hidden_BH, name="both_l0"),
            }
            return loss, log_dict

        return loss, None

    def _step_logs(self) -> dict[str, Any]:
        log_dict = {
            "train/lambda_s": self._lambda_s_scheduler(),
            "train/lambda_f": self._lambda_f_scheduler(),
            "train/lambda_p": self.cfg.lambda_p,
            **self._step_common_logs(),
        }

        if self.step % (self.cfg.log_every_n_steps * 10) == 0:  # type: ignore
            log_dict.update(
                {
                    "media/jumprelu_threshold_distribution": wandb_histogram(
                        self.crosscoder.hidden_activation.log_threshold_H.exp()
                    ),
                }
            )

        return log_dict

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _lambda_f_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_f

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return tanh_sparsity_loss(self.cfg.c, hidden_BH, decoder_norms_H)

    def _pre_act_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return pre_act_loss(self.crosscoder.hidden_activation.log_threshold_H, hidden_BH, decoder_norms_H)
