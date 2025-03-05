from typing import Any

import torch as t
from einops import reduce
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateTrainConfig
from model_diffing.scripts.feb_diff_l1.base_diffing_trainer import BaseDiffingTrainer
from model_diffing.scripts.train_jan_update_crosscoder.trainer import pre_act_loss, tanh_sparsity_loss
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict, l2_norm


class ModelDiffingFebUpdateJumpReLUTrainer(
    BaseDiffingTrainer[JumpReLUModelDiffingFebUpdateTrainConfig, AnthropicJumpReLUActivation]
):
    """
    implementation of https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html but with jumprelu
    loss as described in https://transformer-circuits.pub/2025/january-update/index.html. Expected to be used with
    a JumpReLU crosscoder.
    """

    def _train_step(self, batch_BMD: t.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMD)
        hidden_BH = train_res.get_hidden_BH()

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMD, train_res.recon_acts_BMD)

        # shared features sparsity loss
        shared_dec_norms_Hs = reduce(self.crosscoder._W_dec_shared_HsD, "h_shared dim -> h_shared", l2_norm)
        sparsity_loss_shared = self._tanh_sparsity_loss(train_res.hidden_shared_BHs, shared_dec_norms_Hs)
        lambda_s = self._lambda_s_scheduler()

        # independent features sparsity loss
        indep_dec_norms_Hi = reduce(self.crosscoder._W_dec_indep_HiMD, "h_indep model dim -> h_indep", l2_norm)
        sparsity_loss_indep = self._tanh_sparsity_loss(train_res.hidden_indep_BHi, indep_dec_norms_Hi)
        lambda_f = self._lambda_f_scheduler()

        # pre-activation loss
        dec_norms_H = t.cat([shared_dec_norms_Hs, indep_dec_norms_Hi])
        pre_act_loss = self._pre_act_loss(hidden_BH, dec_norms_H)

        # total loss
        loss = (
            reconstruction_loss  #
            + lambda_s * sparsity_loss_shared
            + lambda_f * sparsity_loss_indep
            + self.cfg.lambda_p * pre_act_loss
        )

        # backward
        loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()
            self._lr_step()

        hidden_BH = train_res.get_hidden_BH()
        self.firing_tracker.add_batch(hidden_BH)

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BMD,
                train_res.recon_acts_BMD,
                ("model", [0, 1]),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/loss": loss.item(),
                "train/lambda_s": lambda_s,
                "train/lambda_f": lambda_f,
                "train/sparsity_loss_shared": sparsity_loss_shared.item(),
                "train/sparsity_loss_indep": sparsity_loss_indep.item(),
                **fvu_dict,
                **get_l0_stats(hidden_BH),
                **self._common_logs(),
            }

            self.wandb_run.log(log_dict, step=self.step)

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _lambda_f_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_f

    def _pre_act_loss(self, hidden_BH: t.Tensor, dec_norms_H: t.Tensor) -> t.Tensor:
        return pre_act_loss(self.crosscoder.hidden_activation.log_threshold_H, hidden_BH, dec_norms_H)

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        return tanh_sparsity_loss(self.cfg.c, hidden_BH, decoder_norms_H)
