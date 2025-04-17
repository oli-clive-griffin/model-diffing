from pathlib import Path
from typing import Any

import einops
import torch

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.relu import ReLUActivation
from crosscode.trainers.crosscoder_wrapper import CrosscoderWrapper
from crosscode.trainers.utils import get_l0_stats
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs, l1_norm, l2_norm


class L1AcausalCrosscoderWrapper(CrosscoderWrapper[ReLUActivation]):
    def __init__(
        self,
        model: ModelHookpointAcausalCrosscoder[ReLUActivation],
        scaling_factors_MP: torch.Tensor,
        hookpoints: list[str],
        model_names: list[str],
        save_dir: Path,
        lambda_s_num_steps: int,
        final_lambda_s: float,
    ):
        super().__init__(
            model,
            scaling_factors_MP,
            hookpoints,
            model_names,
            save_dir,
        )

        self.lambda_s_num_steps = lambda_s_num_steps
        self.final_lambda_s = final_lambda_s

    def _calculate_loss_and_log(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(
            batch.activations_BMPD, train_res.recon_acts_BMPD
        )

        sparsity_loss = sparsity_loss_l1_of_l2s(self.crosscoder.W_dec_LMPD, train_res.latents_BL)

        loss = reconstruction_loss + self._l1_coef_scheduler(step) * sparsity_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                "train/l1_coef": self._l1_coef_scheduler(step),
                **self._get_fvu_dict(batch.activations_BMPD, train_res.recon_acts_BMPD),
                **get_l0_stats(train_res.latents_BL),
            }

            return loss, log_dict

        return loss, None

    def _l1_coef_scheduler(self, step: int) -> float:
        if step < self.lambda_s_num_steps:
            return self.final_lambda_s * step / self.lambda_s_num_steps
        else:
            return self.final_lambda_s


def sparsity_loss_l1_of_l2s(
    W_dec_LMPD: torch.Tensor,
    latents_BL: torch.Tensor,
) -> torch.Tensor:
    assert (latents_BL >= 0).all()
    # think about it like: each latent has a separate projection onto each (model, hookpoint)
    # so we have a separate l2 norm for each (latent, model, hookpoint)
    norms_LX = einops.reduce(W_dec_LMPD, "... d_model -> ...", l2_norm)
    l1_of_norms_L = einops.reduce(norms_LX, "l ... -> l", l1_norm)

    # now we weight the latents by the sum of their output l2 norms
    weighted_latents_BL = latents_BL * l1_of_norms_L
    l1_of_weighted_latents_B = einops.reduce(weighted_latents_BL, "batch latent -> batch", l1_norm)

    return l1_of_weighted_latents_B.mean()
