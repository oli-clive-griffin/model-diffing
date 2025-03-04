from typing import Any

import torch as t
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateTrainConfig
from model_diffing.scripts.feb_diff_l1.base_diffing_trainer import BaseDiffingTrainer
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict, l1_norm, l2_norm


class ModelDiffingFebUpdateJumpReLUTrainer(
    BaseDiffingTrainer[JumpReLUModelDiffingFebUpdateTrainConfig, AnthropicJumpReLUActivation]
):
    """https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html"""

    def _train_step(self, batch_BMD: t.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMD, train_res.recon_acts_BMD)

        # # shared features sparsity loss:
        # decoder_norms_Hs = l2_norm(self.crosscoder._W_dec_shared_HsD, dim=-1)
        # shared_sparsity_loss_BHs = train_res.hidden_BHs * decoder_norms_Hs
        # # take L1 across the features
        # l1_sparsity_loss_B = einsum(shared_sparsity_loss_BHs, "b h_shared -> b")
        # lambda_s = self._lambda_s_scheduler()
        # weighted_shared_sparsity_loss = lambda_s * l1_sparsity_loss_B.mean()

        # # independent features sparsity loss:
        # decoder_norms_HiM = l2_norm(self.crosscoder._W_dec_indep_HiMD, dim=-1)
        # independent_sparsity_loss_BHiM = train_res.hidden_BHi[..., None] * decoder_norms_HiM
        # # take L1 across the features and models
        # independent_sparsity_loss_B = einsum(independent_sparsity_loss_BHiM, "b h_indep m -> b")
        # lambda_f = self._lambda_f_scheduler()
        # weighted_independent_sparsity_loss = lambda_f * independent_sparsity_loss_B.mean()

        # loss = reconstruction_loss + weighted_shared_sparsity_loss + weighted_independent_sparsity_loss

        # backward
        loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()

        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()
            self._lr_step()

        hidden_BH = t.cat([train_res.hidden_BHi, train_res.hidden_BHs], dim=-1)
        self.firing_tracker.add_batch(hidden_BH)

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BMD,
                train_res.recon_acts_BMD,
                ("model", [0, 1]),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/lambda_p": self.cfg.lambda_p,
                "train/loss": loss.item(),
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
