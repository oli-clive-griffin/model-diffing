from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.trainers.firing_tracker import FiringTracker
from crosscode.trainers.trainer import ModelWrapper
from crosscode.trainers.utils import wandb_histogram
from crosscode.utils import get_fvu_dict


class CrossLayerTranscoderWrapper(ModelWrapper):
    def __init__(
        self,
        model: CrossLayerTranscoder[Any],
        scaling_factors_P: torch.Tensor,
        save_dir: Path,
        hookpoints_out: list[str],
    ):
        self.crosscoder = model
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
        train_res = self.crosscoder.forward_train(activations_in_BD)
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
        self.crosscoder.with_folded_scaling_factors(self.scaling_factors_P).save(checkpoint_path)
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

    def parameters(self):
        return self.crosscoder.parameters()


    def before_backward_pass(self) -> None:
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)

