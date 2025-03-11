from functools import partial
from typing import Any

import numpy as np
import torch as t

from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_sliding_window_trainer import BaseSlidingWindowCrosscoderTrainer, BiTokenCCWrapper
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.utils import (
    calculate_reconstruction_loss_summed_MSEs,
    get_fvu_dict,
    l0_norm,
    l1_norm,
    l2_norm,
    weighted_l1_sparsity_loss,
)


class L1SlidingWindowCrosscoderTrainer(BaseSlidingWindowCrosscoderTrainer[ReLUActivation, L1TrainConfig]):
    def _calculate_loss_and_log(
        self,
        batch_BTPD: t.Tensor,
        res: BiTokenCCWrapper.TrainResult,
    ) -> t.Tensor:
        reconstructed_acts_BTPD = t.cat([res.recon_single1_B1PD, res.recon_single2_B1PD], dim=1) + res.recon_double_B2PD
        assert reconstructed_acts_BTPD.shape == batch_BTPD.shape, "fuck"

        # losses
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BTPD, reconstructed_acts_BTPD)

        sparsity_loss_fn = partial(
            weighted_l1_sparsity_loss,
            hookpoint_reduction=l1_norm,  # encourage hookpoint-level sparsity
            model_reduction=l1_norm,  # sum is a noop on a singleton dim
            token_reduction=l2_norm,  # don't want to encourage token-level sparsity
        )

        sparsity_loss_single1 = sparsity_loss_fn(
            W_dec_HTMPD=self.crosscoders.single_cc.W_dec_HXD[:, :, None],
            hidden_BH=res.hidden_single1_BH,
        )
        sparsity_loss_single2 = sparsity_loss_fn(
            W_dec_HTMPD=self.crosscoders.single_cc.W_dec_HXD[:, :, None],
            hidden_BH=res.hidden_single2_BH,
        )
        sparsity_loss_double = sparsity_loss_fn(
            W_dec_HTMPD=self.crosscoders.double_cc.W_dec_HXD[:, :, None],
            hidden_BH=res.hidden_double_BH,
        )
        sparsity_loss = sparsity_loss_single1 + sparsity_loss_single2 + sparsity_loss_double

        loss = reconstruction_loss + self._lambda_s_scheduler() * sparsity_loss

        hidden_B3H = t.cat(
            [res.hidden_single1_BH, res.hidden_double_BH, res.hidden_single2_BH],
            dim=-1,
        )

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            # Instead of building a chart with `get_l0_stats`, we compute and log the values as scalars.
            l0_B = l0_norm(hidden_B3H, dim=-1)
            l0_np = l0_B.detach().cpu().numpy()
            mean_l0 = l0_B.mean().item()
            l0_5, l0_25, l0_75, l0_95 = np.percentile(l0_np, [5, 25, 75, 95])

            fvu_dict = get_fvu_dict(
                batch_BTPD,
                reconstructed_acts_BTPD,
                ("token", [0, 1]),
                ("hookpoint", self.hookpoints),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/lambda_s": self._lambda_s_scheduler(),
                #
                "train/mean_l0_pct": l0_B.mean().item() / hidden_B3H.shape[1],
                # Log grouped l0 statistics as scalars.
                "train/l0/step": self.step,
                "train/l0/5th": l0_5,
                "train/l0/25th": l0_25,
                "train/l0/mean": mean_l0,
                "train/l0/75th": l0_75,
                "train/l0/95th": l0_95,
                #
                "train/loss": loss.item(),
                #
                **fvu_dict,
                #
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            self.wandb_run.log(log_dict, step=self.step)

        return loss

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        if self.step < self.cfg.lambda_s_n_steps:
            return (self.step / self.cfg.lambda_s_n_steps) * self.cfg.final_lambda_s

        return self.cfg.final_lambda_s
