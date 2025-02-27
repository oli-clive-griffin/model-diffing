from typing import Any

import torch
from einops import rearrange
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import calculate_reconstruction_loss, get_fvu_dict, l2_norm, sparsity_loss_l1_of_norms


class AnthropicTransposeInit(InitStrategy[Any]):
    def __init__(self, dec_init_norm: float):
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[Any]) -> None:
        cc.W_dec_HXD.copy_(torch.randn_like(cc.W_dec_HXD))
        W_dec_norm_HX1 = l2_norm(cc.W_dec_HXD, dim=-1, keepdim=True)
        cc.W_dec_HXD.div_(W_dec_norm_HX1)
        cc.W_dec_HXD.mul_(self.dec_init_norm)

        cc.W_enc_XDH.copy_(rearrange(cc.W_dec_HXD.clone(), "h ... d -> ... d h"))

        cc.b_enc_H.zero_()
        cc.b_dec_XD.zero_()


class L1CrosscoderTrainer(BaseModelHookpointTrainer[L1TrainConfig, ReLUActivation]):
    def _train_step(self, batch_BMPD: torch.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMPD)
        self.firing_tracker.add_batch(train_res.hidden_BH)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMPD, train_res.output_BXD)

        W_H1MPD = self.crosscoder.W_dec_HXD[:, None]

        sparsity_loss = sparsity_loss_l1_of_norms(
            W_dec_HTMPD=W_H1MPD,
            hidden_BH=train_res.hidden_BH,
        )
        l1_coef = self._l1_coef_scheduler()
        loss = reconstruction_loss + l1_coef * sparsity_loss

        # backward
        loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()

        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()

        self._lr_step()

        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and self.step % self.cfg.log_every_n_steps == 0
        ):
            fvu_dict = get_fvu_dict(
                batch_BMPD,
                train_res.output_BXD,
                ("model", list(range(self.n_models))),
                ("hookpoint", self.hookpoints),
            )

            log_dict: dict[str, Any] = {
                "train/l1_coef": l1_coef,
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                **fvu_dict,
                **get_l0_stats(train_res.hidden_BH),
                **self._common_logs(),
            }

            self.wandb_run.log(log_dict, step=self.step)

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.lambda_s_n_steps:
            return self.cfg.lambda_s_max * self.step / self.cfg.lambda_s_n_steps
        else:
            return self.cfg.lambda_s_max
