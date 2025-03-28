from typing import Any

import einops
import torch

from model_diffing.models.crosscoder import CrossLayerTranscoder
from model_diffing.trainers.train_l1_crosslayer_trancoder.base_transcoder_trainer import BaseCrossLayerTranscoderTrainer
from model_diffing.trainers.train_l1_crosslayer_trancoder.config import L1TrainConfig
from model_diffing.trainers.utils import get_l0_stats
from model_diffing.utils import calculate_reconstruction_loss_summed_norm_MSEs, l1_norm, l2_norm


class L1CrossLayerTranscoderTrainer(BaseCrossLayerTranscoderTrainer[L1TrainConfig]):
    def _calculate_loss_and_log(
        self,
        train_res: CrossLayerTranscoder.ForwardResult,
        out_BPD: torch.Tensor,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(train_res.output_BPD, out_BPD)

        sparsity_loss = sparsity_loss_l1_of_l2s(self.crosscoder._W_dec_LXoDo, train_res.latents_BL)

        loss = reconstruction_loss + self._l1_coef_scheduler() * sparsity_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(out_BPD, train_res.output_BPD),
                **get_l0_stats(train_res.latents_BL),
            }

            return loss, log_dict

        return loss, None

    def _step_logs(self) -> dict[str, Any]:
        return {
            **super()._step_logs(),
            "train/l1_coef": self._l1_coef_scheduler(),
        }

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.lambda_s_n_steps:
            return self.cfg.final_lambda_s * self.step / self.cfg.lambda_s_n_steps
        else:
            return self.cfg.final_lambda_s


def sparsity_loss_l1_of_l2s(
    W_dec_LXoDo: torch.Tensor,
    latents_BL: torch.Tensor,
) -> torch.Tensor:
    assert (latents_BL >= 0).all()
    # think about it like: each latent has a separate projection onto each (model, hookpoint)
    # so we have a separate l2 norm for each (latent, model, hookpoint)
    norms_LXo = einops.reduce(W_dec_LXoDo, "... d_model -> ...", l2_norm)
    # norms_LXo_ = einx.reduce("... [d_model]", W_dec_LXoDo, l2_norm)
    # assert torch.allclose(norms_LXo, norms_LXo_), 'sanity'

    l1_of_norms_L = einops.reduce(norms_LXo, "l ... -> l", l1_norm)
    # l1_of_norms_L_ = einx.reduce("l [...]", norms_LXo, l1_norm)
    # assert torch.allclose(l1_of_norms_L, l1_of_norms_L_), 'sanity'

    # now we weight the latents by the sum of their output l2 norms
    weighted_latents_BL = latents_BL * l1_of_norms_L
    l1_of_weighted_latents_B = einops.reduce(weighted_latents_BL, "batch latent -> batch", l1_norm)

    return l1_of_weighted_latents_B.mean()
