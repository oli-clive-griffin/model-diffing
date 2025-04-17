from pathlib import Path
from typing import Any

import torch

from crosscode.models.activations.topk import TopkActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.trainers.crosslayer_transcoder_wrapper import CrossLayerTranscoderWrapper
from crosscode.trainers.topk_crosscoder.trainer import aux_loss
from crosscode.trainers.utils import get_l0_stats
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs, not_none


class TopkCrossLayerTranscoderWrapper(CrossLayerTranscoderWrapper):
    def __init__(
        self,
        model: CrossLayerTranscoder[TopkActivation],
        scaling_factors_P: torch.Tensor,
        save_dir: Path,
        hookpoints_out: list[str],
        lambda_aux: float,
        k_aux: int,
        dead_latents_threshold_n_examples: int,
    ):
        super().__init__(
            model,
            scaling_factors_P,
            save_dir,
            hookpoints_out,
        )
        self.lambda_aux = lambda_aux
        self.k_aux = k_aux
        self.dead_latents_threshold_n_examples = dead_latents_threshold_n_examples

    def _calculate_loss_and_log(
        self,
        step: int,
        target_BPD: torch.Tensor,
        train_res: CrossLayerTranscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(train_res.output_BPD, target_BPD)
        aux_loss = self.aux_loss(target_BPD, train_res)
        loss = reconstruction_loss + self.lambda_aux * aux_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/aux_loss": aux_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(target_BPD, train_res.output_BPD),
                **get_l0_stats(train_res.latents_BL),
            }
            return loss, log_dict

        return loss, None

    def aux_loss(self, target_BPD: torch.Tensor, train_res: CrossLayerTranscoder.ForwardResult) -> torch.Tensor:
        """train to reconstruct the error with the topk dead latents"""
        return aux_loss(
            pre_activations_BL=train_res.pre_activations_BL,
            dead_features_mask_L=self.firing_tracker.tokens_since_fired_L > self.dead_latents_threshold_n_examples,
            k_aux=not_none(self.k_aux),
            decode_BXD=self.crosscoder.decode_BXoDo,
            error_BXD=target_BPD - train_res.output_BPD,
        )
