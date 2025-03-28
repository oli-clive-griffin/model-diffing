from typing import Any

import torch as t

from model_diffing.models.activations.jumprelu import AnthropicSTEJumpReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_diffing_trainer import BaseDiffingTrainer
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateTrainConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import pre_act_loss, tanh_sparsity_loss
from model_diffing.scripts.utils import get_l0_stats, wandb_histogram
from model_diffing.utils import calculate_reconstruction_loss_summed_norm_MSEs, get_summed_decoder_norms_L


class ModelDiffingFebUpdateJumpReLUTrainer(
    BaseDiffingTrainer[JumpReLUModelDiffingFebUpdateTrainConfig, AnthropicSTEJumpReLUActivation]
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
        log: bool,
    ) -> tuple[t.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(batch_BMD, train_res.recon_acts_BXD)

        decoder_norms_L = get_summed_decoder_norms_L(self.crosscoder._W_dec_LXoDo)
        decoder_norms_shared_Ls = decoder_norms_L[: self.n_shared_latents]
        decoder_norms_indep_Li = decoder_norms_L[self.n_shared_latents :]

        hidden_shared_BLs = train_res.latents_BL[:, : self.n_shared_latents]
        hidden_indep_BLi = train_res.latents_BL[:, self.n_shared_latents :]

        # shared features sparsity loss
        tanh_sparsity_loss_shared = self._tanh_sparsity_loss(hidden_shared_BLs, decoder_norms_shared_Ls)
        lambda_s = self._lambda_s_scheduler()

        # independent features sparsity loss
        tanh_sparsity_loss_indep = self._tanh_sparsity_loss(hidden_indep_BLi, decoder_norms_indep_Li)
        lambda_f = self._lambda_f_scheduler()

        # pre-activation loss on all features
        pre_act_loss = self._pre_act_loss(train_res.latents_BL, decoder_norms_L)

        # total loss
        loss = (
            reconstruction_loss  #
            + lambda_s * tanh_sparsity_loss_shared
            + lambda_f * tanh_sparsity_loss_indep
            + self.cfg.lambda_p * pre_act_loss
        )

        if log:
            hidden_shared_BLs = train_res.latents_BL[:, : self.n_shared_latents]
            hidden_indep_BLi = train_res.latents_BL[:, self.n_shared_latents :]

            log_dict: dict[str, float] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss_shared": tanh_sparsity_loss_shared.item(),
                "train/tanh_sparsity_loss_indep": tanh_sparsity_loss_indep.item(),
                "train/pre_act_loss": pre_act_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(batch_BMD, train_res.recon_acts_BXD),
                **get_l0_stats(hidden_shared_BLs, name="shared_l0"),
                **get_l0_stats(hidden_indep_BLi, name="indep_l0"),
                **get_l0_stats(train_res.latents_BL, name="both_l0"),
            }
            return loss, log_dict

        return loss, None

    def _step_logs(self) -> dict[str, Any]:
        log_dict: dict[str, Any] = {
            **super()._step_logs(),
            "train/lambda_s": self._lambda_s_scheduler(),
            "train/lambda_f": self._lambda_f_scheduler(),
            "train/lambda_p": self.cfg.lambda_p,
        }

        if (
            self.cfg.log_every_n_steps is not None
            and self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0
        ):
            jr_threshold_hist = wandb_histogram(self.crosscoder.activation_fn.log_threshold_L.exp())
            log_dict.update({"media/jr_threshold": jr_threshold_hist})

        return log_dict

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _lambda_f_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_f

    def _tanh_sparsity_loss(self, hidden_BL: t.Tensor, decoder_norms_L: t.Tensor) -> t.Tensor:
        return tanh_sparsity_loss(self.cfg.c, hidden_BL, decoder_norms_L)

    def _pre_act_loss(self, hidden_BL: t.Tensor, decoder_norms_L: t.Tensor) -> t.Tensor:
        return pre_act_loss(self.crosscoder.activation_fn.log_threshold_L, hidden_BL, decoder_norms_L)
