from abc import abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from crosscode.data.activations_dataloader import (
    ActivationsDataloader,
    ModelHookpointActivationsBatch,
    ModelHookpointActivationsDataloader,
)
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.activation_function import ActivationFunction
from crosscode.trainers.base_trainer import BaseTrainer
from crosscode.trainers.config_common import BaseTrainConfig
from crosscode.trainers.utils import create_cosine_sim_and_relative_norm_histograms, wandb_histogram
from crosscode.trainers.wandb_utils.main import create_checkpoint_artifact
from crosscode.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseModelHookpointAcausalTrainer(
    BaseTrainer[TConfig, ModelHookpointAcausalCrosscoder[TAct], ModelHookpointActivationsBatch]
):
    activations_dataloader: ModelHookpointActivationsDataloader

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: ActivationsDataloader[ModelHookpointActivationsBatch],
        model: ModelHookpointAcausalCrosscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
    ):
        super().__init__(cfg, activations_dataloader, model, wandb_run, device, save_dir)

        assert self.activations_dataloader.n_models == self.model.n_models, (
            "expected the number of models to be the same between the activations dataloader and the crosscoder"
        )
        self.n_models = self.activations_dataloader.n_models

        assert self.activations_dataloader.n_hookpoints == self.model.n_hookpoints, (
            "expected the number of hookpoints to be the same between the activations dataloader and the crosscoder"
        )
        self.hookpoints = self.activations_dataloader.hookpoints

    def run_batch(
        self, batch: ModelHookpointActivationsBatch, log: bool
    ) -> tuple[torch.Tensor, dict[str, float] | None, int]:
        train_res = self.model.forward_train(batch.activations_BMPD.to(self.device))
        self.firing_tracker.add_batch(train_res.latents_BL)
        loss, log_dict = self._calculate_loss_and_log(batch.activations_BMPD, train_res, log=log)
        return loss, log_dict, batch.activations_BMPD.shape[0]

    def _maybe_save_model(self) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            scaling_factors_MP = self.activations_dataloader.get_scaling_factors().to(self.device)
            self.model.with_folded_scaling_factors(scaling_factors_MP).save(checkpoint_path)

            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def _step_logs(self) -> dict[str, Any]:
        log_dict = super()._step_logs()
        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.tokens_since_fired_L)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})
            if self.n_models == 2:
                W_dec_LMPD = self.model.W_dec_LMPD.detach().cpu()
                for p, hookpoint in enumerate(self.hookpoints):
                    W_dec_LMD = W_dec_LMPD[:, :, p]
                    relative_decoder_norms_plot, shared_features_cosine_sims_plot = (
                        create_cosine_sim_and_relative_norm_histograms(W_dec_LMD)
                    )
                    if relative_decoder_norms_plot is not None:
                        log_dict.update(
                            {
                                f"media/relative_decoder_norms_{hookpoint}": relative_decoder_norms_plot,
                            }
                        )
                    if shared_features_cosine_sims_plot is not None:
                        log_dict.update(
                            {
                                f"media/shared_features_cosine_sims_{hookpoint}": shared_features_cosine_sims_plot,
                            }
                        )

        return log_dict

    def _get_fvu_dict(self, batch_BMPD: torch.Tensor, recon_acts_BMPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            batch_BMPD,
            recon_acts_BMPD,
            ("model", ["0", "1"]),
            ("hookpoint", self.hookpoints),
        )
