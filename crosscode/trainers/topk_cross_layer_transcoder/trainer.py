from typing import Any

import torch

from crosscode.models.activations.topk import TopkActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.trainers.base_crosslayer_transcoder_trainer import BaseCrossLayerTranscoderTrainer
from crosscode.trainers.topk_crosscoder.config import TopKTrainConfig
from crosscode.trainers.topk_crosscoder.trainer import aux_loss
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs, not_none


class TopkCrossLayerTranscoderTrainer(BaseCrossLayerTranscoderTrainer[TopKTrainConfig, TopkActivation]):
    def _calculate_loss_and_log(
        self,
        train_res: CrossLayerTranscoder.ForwardResult,
        target_BPD: torch.Tensor,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(train_res.output_BPD, target_BPD)
        aux_loss = self.aux_loss(target_BPD, train_res)
        loss = reconstruction_loss + self.cfg.lambda_aux * aux_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/aux_loss": aux_loss.item(),
                "train/loss": loss.item(),
                **self._get_fvu_dict(target_BPD, train_res.output_BPD),
            }
            return loss, log_dict

        return loss, None

    def aux_loss(self, target_BPD: torch.Tensor, train_res: CrossLayerTranscoder.ForwardResult) -> torch.Tensor:
        """train to reconstruct the error with the topk dead latents"""
        return aux_loss(
            pre_activations_BL=train_res.pre_activations_BL,
            dead_features_mask_L=self.firing_tracker.tokens_since_fired_L > self.cfg.dead_latents_threshold_n_examples,
            k_aux=not_none(self.cfg.k_aux),
            decode_BXD=self.model.decode_BXoDo,
            error_BXD=target_BPD - train_res.output_BPD,
        )
