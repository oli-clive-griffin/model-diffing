from abc import abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from crosscoding.data.activations_dataloader import ModelHookpointActivationsBatch, ModelHookpointActivationsDataloader
from crosscoding.dims import CrosscodingDim
from crosscoding.models.activations.activation_function import ActivationFunction
from crosscoding.models.sparse_coders import CrossLayerTranscoder
from crosscoding.trainers.base_acausal_trainer import BaseTrainer
from crosscoding.trainers.config_common import BaseTrainConfig
from crosscoding.trainers.wandb_utils.main import create_checkpoint_artifact
from crosscoding.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseCrossLayerTranscoderTrainer(
    Generic[TConfig, TAct],
    BaseTrainer[TConfig, CrossLayerTranscoder[TAct], ModelHookpointActivationsBatch],
):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: ModelHookpointActivationsDataloader,
        crosscoder: CrossLayerTranscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
        out_layers_names: list[str],
    ):
        super().__init__(cfg, activations_dataloader, crosscoder, wandb_run, device, save_dir)
        self.out_layers_names = out_layers_names
        self.out_layers_dim = CrosscodingDim(name="out_layer", index_labels=out_layers_names)

        dl_cc_dims = self.activations_dataloader.get_crosscoding_dims()
        assert len(dl_cc_dims["model"]) == 1
        assert len(dl_cc_dims["hookpoint"]) == len(self.out_layers_dim) + 1

    def run_batch(
        self, batch: ModelHookpointActivationsBatch, log: bool
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        batch_BMPD = batch.activations_BMPD
        assert batch_BMPD.shape[1] == 1, "we must have one model"
        assert batch_BMPD.shape[2] == len(self.out_layers_dim) + 1, "we must have one more hookpoint than out layers"
        in_BD = batch_BMPD[:, 0, 0]
        out_BPD = batch_BMPD[:, 0, 1:]

        train_res = self.crosscoder.forward_train(in_BD)

        self.firing_tracker.add_batch(train_res.latents_BL)

        return self._calculate_loss_and_log(train_res, out_BPD, log=log)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        train_res: CrossLayerTranscoder.ForwardResult,
        out_BPD: torch.Tensor,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def _get_fvu_dict(self, y_BPD: torch.Tensor, recon_y_BPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            y_BPD,
            recon_y_BPD,
            ("hookpoint", self.out_layers_dim.index_labels),
        )

    def _maybe_save_model(self) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            # We know this is (model, hookpoint) because it's a ModelHookpointActivationsDataloader
            scaling_factors_MP = self.activations_dataloader.get_scaling_factors()
            assert scaling_factors_MP.shape[1] == 2, "expected the scaling factors to have one model only"
            scaling_factor_in_ = scaling_factors_MP[0, 0]  # shape ()
            scaling_factors_out_P = scaling_factors_MP[0, 1:]  # shape (P,)
            self.crosscoder.with_folded_scaling_factors(scaling_factor_in_, scaling_factors_out_P).save(
                checkpoint_path
            )

            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)
