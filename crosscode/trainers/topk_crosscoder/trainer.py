from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.topk import BatchTopkActivation, GroupMaxActivation, TopkActivation, topk_activation
from crosscode.trainers.crosscoder_wrapper import CrosscoderWrapper
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs


class TopKAcausalCrosscoderWrapper(CrosscoderWrapper[TopkActivation | BatchTopkActivation | GroupMaxActivation]):
    def __init__(
        self,
        model: ModelHookpointAcausalCrosscoder[TopkActivation | BatchTopkActivation | GroupMaxActivation],
        hookpoints: list[str],
        model_names: list[str],
        save_dir: Path,
        scaling_factors_MP: torch.Tensor,
        lambda_aux: float,
        k_aux: int,
        dead_latents_threshold_n_examples: int,
    ):
        super().__init__(
            model,
            scaling_factors_MP,
            hookpoints,
            model_names,
            save_dir,
        )

        self.lambda_aux = lambda_aux
        self.k_aux = k_aux
        self.dead_latents_threshold_n_examples = dead_latents_threshold_n_examples

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
        aux_loss = self._aux_loss(batch.activations_BMPD, train_res)
        loss = reconstruction_loss + self.lambda_aux * aux_loss

        if log:
            log_dict: dict[str, Any] = {
                "train/loss": loss.item(),
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/aux_loss": aux_loss.item(),
                **self._get_fvu_dict(batch.activations_BMPD, train_res.recon_acts_BMPD),
            }

            return loss, log_dict

        return loss, None

    def _aux_loss(
        self, batch_BMPD: torch.Tensor, train_res: ModelHookpointAcausalCrosscoder.ForwardResult
    ) -> torch.Tensor:
        """train to reconstruct the error with the topk dead latents"""
        return aux_loss(
            pre_activations_BL=train_res.pre_activations_BL,
            dead_features_mask_L=self.firing_tracker.tokens_since_fired_L > self.dead_latents_threshold_n_examples,
            k_aux=self.k_aux,
            decode_BXD=self.crosscoder.decode_BMPD,
            error_BXD=batch_BMPD - train_res.recon_acts_BMPD,
        )

    def save(self, step: int) -> Path:
        checkpoint_path = self.save_dir / f"step_{step}"
        self.crosscoder.with_folded_scaling_factors(self.scaling_factors_MP).save(checkpoint_path)
        return checkpoint_path


def aux_loss(
    pre_activations_BL: torch.Tensor,
    dead_features_mask_L: torch.Tensor,
    k_aux: int,
    decode_BXD: Callable[[torch.Tensor], torch.Tensor],
    error_BXD: torch.Tensor,
) -> torch.Tensor:
    if (topk_aux_output := topk_dead_latents(pre_activations_BL, dead_features_mask_L, k_aux)) is None:
        return torch.tensor(0.0, device=pre_activations_BL.device)

    aux_latents_BL, n_latents_used = topk_aux_output

    # If there's less than `k_aux` dead features, it's harder to reconstruct the error. so scale down the loss.
    aux_loss_scale = n_latents_used / k_aux

    # try to reconstruct the error with the topk dead latents
    error_recon_mse = calculate_reconstruction_loss_summed_norm_MSEs(decode_BXD(aux_latents_BL), error_BXD)
    return error_recon_mse * aux_loss_scale


def topk_dead_latents(
    pre_activations_BL: torch.Tensor,
    dead_features_mask_L: torch.Tensor,
    k_aux: int,
) -> tuple[torch.Tensor, int] | None:
    n_dead = int(dead_features_mask_L.sum())
    if n_dead == 0:
        return None

    dead_latents_BL = pre_activations_BL * dead_features_mask_L

    # we only need to actually do the topk operation if there are more dead features than k_aux_base
    aux_latents_BL = topk_activation(dead_latents_BL, k_aux) if n_dead > k_aux else dead_latents_BL
    n_latents_used = min(n_dead, k_aux)

    return aux_latents_BL, n_latents_used
