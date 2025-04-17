from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.activation_function import ActivationFunction
from crosscode.trainers.firing_tracker import FiringTracker
from crosscode.trainers.trainer import ModelWrapper
from crosscode.trainers.utils import create_cosine_sim_and_relative_norm_histograms, wandb_histogram
from crosscode.utils import get_fvu_dict

TActivation = TypeVar("TActivation", bound=ActivationFunction)


class CrosscoderWrapper(Generic[TActivation], ModelWrapper):
    def __init__(
        self,
        model: ModelHookpointAcausalCrosscoder[TActivation],
        scaling_factors_MP: torch.Tensor,
        hookpoints: list[str],
        model_names: list[str],
        save_dir: Path,
    ):
        self.model = model
        self.scaling_factors_MP = scaling_factors_MP
        self.hookpoints = hookpoints
        self.model_names = model_names
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.firing_tracker = FiringTracker(activation_size=model.n_latents, device=model.device)

    def run_batc(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        train_res = self.model.forward_train(batch.activations_BMPD)
        self.firing_tracker.add_batch(train_res.latents_BL)
        return self._calculate_loss_and_log(step, batch, train_res, log)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def expensive_logs(self) -> dict[str, Any]:
        log_dict: dict[str, Any] = {
            "media/tokens_since_fired": wandb_histogram(self.firing_tracker.tokens_since_fired_L)
        }

        if self.model.b_enc_L is not None:
            log_dict["b_enc_values"] = wandb_histogram(self.model.b_enc_L)

        if self.model.n_models == 2:
            W_dec_LMPD = self.model.W_dec_LMPD.detach()  # .cpu()
            for p, hookpoint in enumerate(self.hookpoints):
                W_dec_LMD = W_dec_LMPD[:, :, p]
                relative_decoder_norms_plot, shared_features_cosine_sims_plot = (
                    create_cosine_sim_and_relative_norm_histograms(W_dec_LMD)
                )
                if relative_decoder_norms_plot is not None:
                    log_dict[f"media/relative_decoder_norms_{hookpoint}"] = relative_decoder_norms_plot
                if shared_features_cosine_sims_plot is not None:
                    log_dict[f"media/shared_features_cosine_sims_{hookpoint}"] = shared_features_cosine_sims_plot

        return log_dict

    def _get_fvu_dict(self, batch_BMPD: torch.Tensor, recon_acts_BMPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            batch_BMPD,
            recon_acts_BMPD,
            ("model", self.model_names),
            ("hookpoint", self.hookpoints),
        )

    def save(self, step: int) -> Path:
        checkpoint_path = self.save_dir / f"step_{step}"
        self.model.with_folded_scaling_factors(self.scaling_factors_MP).save(checkpoint_path)
        return checkpoint_path
