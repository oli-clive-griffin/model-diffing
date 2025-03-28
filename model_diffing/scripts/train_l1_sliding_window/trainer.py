from functools import partial
from typing import Any

import torch as t

from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_sliding_window_trainer import BaseSlidingWindowCrosscoderTrainer, BiTokenCCWrapper
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import (
    calculate_reconstruction_loss_summed_norm_MSEs,
    l1_norm,
    l2_norm,
    weighted_l1_sparsity_loss,
)


class L1SlidingWindowCrosscoderTrainer(BaseSlidingWindowCrosscoderTrainer[ReLUActivation, L1TrainConfig]):
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

        sparsity_loss_fn = partial(
            weighted_l1_sparsity_loss,
            hookpoint_reduction=l1_norm,  # encourage hookpoint-level sparsity
            model_reduction=l1_norm,  # sum is a noop on a singleton dim
            token_reduction=l2_norm,  # don't want to encourage token-level sparsity
        )

        sparsity_loss_single1 = sparsity_loss_fn(
            W_dec_LTMPD=self.crosscoders.single_cc._W_dec_LXoDo[:, :, None],
            latents_BL=res.hidden_single1_BL,
        )
        sparsity_loss_single2 = sparsity_loss_fn(
            W_dec_LTMPD=self.crosscoders.single_cc._W_dec_LXoDo[:, :, None],
            latents_BL=res.hidden_single2_BL,
        )
        sparsity_loss_double = sparsity_loss_fn(
            W_dec_LTMPD=self.crosscoders.double_cc._W_dec_LXoDo[:, :, None],
            latents_BL=res.hidden_double_BL,
        )
        sparsity_loss = sparsity_loss_single1 + sparsity_loss_single2 + sparsity_loss_double

        loss = reconstruction_loss + self._lambda_s_scheduler() * sparsity_loss

        hidden_B3H = t.cat(
            [res.hidden_single1_BL, res.hidden_double_BL, res.hidden_single2_BL],
            dim=-1,
        )

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                **get_l0_stats(hidden_B3H),
                **self._get_fvu_dict(batch_BTPD, reconstructed_acts_BTPD),
            }

            return loss, log_dict

        return loss, None

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        if self.step < self.cfg.lambda_s_n_steps:
            return (self.step / self.cfg.lambda_s_n_steps) * self.cfg.final_lambda_s

        return self.cfg.final_lambda_s

    def _step_logs(self) -> dict[str, Any]:
        return {
            **super()._step_logs(),
            "train/lambda_s": self._lambda_s_scheduler(),
        }
