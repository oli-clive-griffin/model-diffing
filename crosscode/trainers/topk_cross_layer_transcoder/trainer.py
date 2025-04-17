from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.activations.topk import TopkActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.trainers.firing_tracker import FiringTracker
from crosscode.trainers.topk_crosscoder.trainer import aux_loss
from crosscode.trainers.trainer import ModelWrapper
from crosscode.trainers.utils import get_l0_stats, wandb_histogram
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs, get_fvu_dict, not_none


class CrossLayerTranscoderWrapper(ModelWrapper):
    def __init__(
        self,
        model: CrossLayerTranscoder[Any],
        scaling_factors_P: torch.Tensor,
        save_dir: Path,
        hookpoints_out: list[str],
    ):
        self.model = model
        self.hookpoints_out = hookpoints_out
        self.scaling_factors_P = scaling_factors_P.to(model.device)

        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.firing_tracker = FiringTracker(activation_size=model.n_latents, device=model.device)

    def run_batch(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        _, n_models, n_hookpoints, _ = batch.activations_BMPD.shape
        assert n_models == 1, "we must have one model"
        assert n_hookpoints == len(self.hookpoints_out) + 1, "we must have one more hookpoint than out layers"
        activations_in_BD = batch.activations_BMPD[:, 0, 0]
        target_BPD = batch.activations_BMPD[:, 0, 1:]
        train_res = self.model.forward_train(activations_in_BD)
        self.firing_tracker.add_batch(train_res.latents_BL)
        return self._calculate_loss_and_log(step, target_BPD, train_res, log)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        step: int,
        target_BPD: torch.Tensor,
        train_res: CrossLayerTranscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def save(self, step: int) -> Path:
        checkpoint_path = self.save_dir / f"step_{step}"
        self.model.with_folded_scaling_factors(self.scaling_factors_P).save(checkpoint_path)
        return checkpoint_path

    def expensive_logs(self) -> dict[str, Any]:
        return {
            "train/tokens_since_fired": wandb_histogram(self.firing_tracker.tokens_since_fired_L),
            # Add a pl.img of layer, latent groupings?
        }

    def _get_fvu_dict(self, batch_BPD: torch.Tensor, recon_acts_BPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            batch_BPD,
            recon_acts_BPD,
            ("hookpoint", self.hookpoints_out),
        )


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
            decode_BXD=self.model.decode_BXoDo,
            error_BXD=target_BPD - train_res.output_BPD,
        )
