from typing import Any

import torch
from einops import rearrange

from model_diffing.models import InitStrategy
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.scripts.utils import get_l0_stats
from model_diffing.utils import (
    calculate_reconstruction_loss_summed_MSEs,
    get_fvu_dict,
    random_direction_init_,
    sparsity_loss_l1_of_norms,
)


class AnthropicTransposeInit(InitStrategy[AcausalCrosscoder[Any]]):
    def __init__(self, dec_init_norm: float):
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[Any]) -> None:
        random_direction_init_(cc.W_dec_HXD, self.dec_init_norm)

        cc.W_enc_XDH.copy_(rearrange(cc.W_dec_HXD.clone(), "h ... -> ... h"))

        cc.b_enc_H.zero_()
        cc.b_dec_XD.zero_()


class L1CrosscoderTrainer(BaseModelHookpointTrainer[L1TrainConfig, ReLUActivation]):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
    ) -> torch.Tensor:
        reconstruction_loss = calculate_reconstruction_loss_summed_MSEs(batch_BMPD, train_res.recon_acts_BXD)

        sparsity_loss = sparsity_loss_l1_of_norms(
            W_dec_HTMPD=self.crosscoder.W_dec_HXD[:, None],
            hidden_BH=train_res.hidden_BH,
        )

        l1_coef = self._l1_coef_scheduler()

        loss = reconstruction_loss + l1_coef * sparsity_loss

        if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
            fvu_dict = get_fvu_dict(
                batch_BMPD,
                train_res.recon_acts_BXD,
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

        return loss

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.lambda_s_n_steps:
            return self.cfg.final_lambda_s * self.step / self.cfg.lambda_s_n_steps
        else:
            return self.cfg.final_lambda_s
