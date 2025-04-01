from typing import Any

import torch

from crosscoding.models.activations.topk import BatchTopkActivation, GroupMaxActivation, TopkActivation
from crosscoding.models.sparse_coders import ModelHookpointAcausalCrosscoder
from crosscoding.trainers.base_diffing_trainer import BaseFebUpdateDiffingTrainer
from crosscoding.trainers.train_topk_crosscoder.config import TopKTrainConfig
from crosscoding.trainers.train_topk_crosscoder.trainer import aux_loss
from crosscoding.utils import calculate_reconstruction_loss_summed_norm_MSEs, not_none


class TopKFebUpdateDiffingTrainer(
    BaseFebUpdateDiffingTrainer[TopKTrainConfig, TopkActivation | GroupMaxActivation | BatchTopkActivation]
):
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(batch_BMPD, train_res.recon_acts_BMPD)
        aux_loss = self.aux_loss(batch_BMPD, train_res)
        loss = reconstruction_loss + self.cfg.lambda_aux * aux_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/loss": loss.item(),
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/aux_loss": aux_loss.item(),
                **self._get_fvu_dict(batch_BMPD, train_res.recon_acts_BMPD),
            }

            return reconstruction_loss, log_dict

        return reconstruction_loss, None

    def aux_loss(
        self, batch_BMPD: torch.Tensor, train_res: ModelHookpointAcausalCrosscoder.ForwardResult
    ) -> torch.Tensor:
        """train to reconstruct the error with the topk dead latents"""
        return aux_loss(
            pre_activations_BL=train_res.pre_activations_BL,
            dead_features_mask_L=self.firing_tracker.tokens_since_fired_L > self.cfg.dead_latents_threshold_n_examples,
            k_aux=not_none(self.cfg.k_aux),
            decode_BXD=self.crosscoder.decode_BMPD,
            error_BXD=batch_BMPD - train_res.recon_acts_BMPD,
        )