from pathlib import Path
from typing import Any

import torch

from crosscode.models.activations.relu import ReLUActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.trainers.crosslayer_transcoder_wrapper import CrossLayerTranscoderWrapper
from crosscode.trainers.l1_crosscoder.trainer import sparsity_loss_l1_of_l2s
from crosscode.trainers.utils import get_l0_stats
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs


class L1CrossLayerTranscoderWrapper(CrossLayerTranscoderWrapper):
    def __init__(
        self,
        model: CrossLayerTranscoder[ReLUActivation],
        scaling_factors_P: torch.Tensor,
        hookpoints_out: list[str],
        save_dir: Path,
        lambda_s_num_steps: int,
        final_lambda_s: float,
    ):
        super().__init__(
            model,
            scaling_factors_P,
            save_dir,
            hookpoints_out,
        )
        self.lambda_s_num_steps = lambda_s_num_steps
        self.final_lambda_s = final_lambda_s

    def _calculate_loss_and_log(
        self,
        step: int,
        target_BPD: torch.Tensor,
        train_res: CrossLayerTranscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(train_res.output_BPD, target_BPD)

        sparsity_loss = sparsity_loss_l1_of_l2s(self.crosscoder.W_dec_LPD, train_res.latents_BL)

        lambda_s = self._l1_coef_scheduler(step)
        loss = reconstruction_loss + lambda_s * sparsity_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                "train/lambda_s": lambda_s,
                **self._get_fvu_dict(target_BPD, train_res.output_BPD),
                **get_l0_stats(train_res.latents_BL),
            }

            return loss, log_dict

        return loss, None

    def _l1_coef_scheduler(self, step: int) -> float:
        if step < self.lambda_s_num_steps:
            return self.final_lambda_s * step / self.lambda_s_num_steps
        else:
            return self.final_lambda_s
