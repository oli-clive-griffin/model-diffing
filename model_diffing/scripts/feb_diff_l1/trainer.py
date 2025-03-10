from typing import Any

import torch as t
from einops import reduce
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_diffing_trainer import BaseDiffingTrainer
from model_diffing.scripts.feb_diff_l1.config import L1ModelDiffingFebUpdateTrainConfig
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict, l1_norm, l2_norm


class ModelDiffingFebUpdateL1Trainer(BaseDiffingTrainer[L1ModelDiffingFebUpdateTrainConfig, ReLUActivation]):
    """
    implementation of https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html with l1 sparsity loss
    """

    def _train_step(self, batch_BMD: t.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMD, train_res.recon_acts_BMD)

        # shared features sparsity loss:
        decoder_norms_Hs = l2_norm(self.crosscoder._W_dec_shared_m0_HsD, dim=-1)
        sparsity_loss_shared_BHs = train_res.hidden_shared_BHs * decoder_norms_Hs
        sparsity_loss_shared = reduce(sparsity_loss_shared_BHs, "b h_shared -> b", l1_norm).mean()
        lambda_s = self._lambda_s_scheduler()
        scaled_sparsity_loss_shared = lambda_s * sparsity_loss_shared

        # indep features sparsity loss:
        decoder_norms_HiM = l2_norm(self.crosscoder._W_dec_indep_HiMD, dim=-1)
        sparsity_loss_indep_BHiM = train_res.hidden_indep_BHi[..., None] * decoder_norms_HiM
        sparsity_loss_indep = reduce(sparsity_loss_indep_BHiM, "b h_indep m -> b", l1_norm).mean()  # across models too!
        lambda_f = self._lambda_f_scheduler()
        scaled_sparsity_loss_indep = lambda_f * sparsity_loss_indep

        loss = (
            reconstruction_loss  #
            + scaled_sparsity_loss_shared
            + scaled_sparsity_loss_indep
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
                "train/scaled_sparsity_loss_shared": scaled_sparsity_loss_shared.item(),
                "train/scaled_sparsity_loss_indep": scaled_sparsity_loss_indep.item(),
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
