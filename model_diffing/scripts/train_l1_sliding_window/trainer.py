from functools import partial
from typing import Any

import numpy as np
import torch as t
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.scripts.train_l1_sliding_window.base_sliding_window_trainer import BaseSlidingWindowCrosscoderTrainer
from model_diffing.utils import (
    calculate_fvu_X,
    calculate_reconstruction_loss,
    get_fvu_dict,
    l0_norm,
    l1_norm,
    l2_norm,
    weighted_l1_sparsity_loss,
)


class L1SlidingWindowCrosscoderTrainer(BaseSlidingWindowCrosscoderTrainer[ReLUActivation, L1TrainConfig]):
    def _train_step(self, batch_BTPD: t.Tensor) -> None:
        self.optimizer.zero_grad()

        # Forward pass
        res = self.crosscoders.forward_train(batch_BTPD)
        reconstructed_acts_BTPD = t.cat([res.recon_B1PD_single1, res.recon_B1PD_single2], dim=1) + res.recon_B2PD_double
        assert reconstructed_acts_BTPD.shape == batch_BTPD.shape, "fuck"

        reconstruction_loss = calculate_reconstruction_loss(batch_BTPD, reconstructed_acts_BTPD)

        sparsity_loss_fn = partial(
            weighted_l1_sparsity_loss,
            hookpoint_reduction=l1_norm,  # encourage hookpoint-level sparsity
            model_reduction=l1_norm,  # sum is a noop on a singleton dim
            token_reduction=l2_norm,  # don't want to encourage token-level sparsity
        )

        sparsity_loss_single1 = sparsity_loss_fn(
            W_dec_HTMPD=self.crosscoders.single_cc.W_dec_HXD[:, :, None],
            hidden_BH=res.hidden_BH_single1,
        )
        sparsity_loss_single2 = sparsity_loss_fn(
            W_dec_HTMPD=self.crosscoders.single_cc.W_dec_HXD[:, :, None],
            hidden_BH=res.hidden_BH_single2,
        )
        sparsity_loss_double = sparsity_loss_fn(
            W_dec_HTMPD=self.crosscoders.double_cc.W_dec_HXD[:, :, None],
            hidden_BH=res.hidden_BH_double,
        )
        sparsity_loss = sparsity_loss_single1 + sparsity_loss_single2 + sparsity_loss_double

        loss = reconstruction_loss + self._lambda_s_scheduler() * sparsity_loss

        # Backward pass
        loss.backward()
        clip_grad_norm_(self.crosscoders.parameters(), 1.0)
        self.optimizer.step()

        # Update learning rate according to scheduler.
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        hidden_B3H = t.cat(
            [res.hidden_BH_single1, res.hidden_BH_double, res.hidden_BH_single2],
            dim=-1,
        )

        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and (self.step + 1) % self.cfg.log_every_n_steps == 0
        ):
            # Instead of building a chart with `get_l0_stats`, we compute and log the values as scalars.
            l0_B = l0_norm(hidden_B3H, dim=-1)
            l0_np = l0_B.detach().cpu().numpy()
            mean_l0 = l0_B.mean().item()
            l0_5, l0_25, l0_75, l0_95 = np.percentile(l0_np, [5, 25, 75, 95])

            fvu_dict = get_fvu_dict(
                calculate_fvu_X(batch_BTPD, reconstructed_acts_BTPD),
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

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        if self.step < self.cfg.lambda_s_n_steps:
            return (self.step / self.cfg.lambda_s_n_steps) * self.cfg.lambda_s_max

        return self.cfg.lambda_s_max
