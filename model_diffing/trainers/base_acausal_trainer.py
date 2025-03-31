from abc import abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from model_diffing.data.base_activations_dataloader import BaseActivationsDataloader, CrosscodingDims
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.trainers.base_trainer import BaseTrainer
from model_diffing.trainers.config_common import BaseTrainConfig
from model_diffing.trainers.utils import create_cosine_sim_and_relative_norm_histograms, wandb_histogram
from model_diffing.trainers.wandb_utils.main import create_checkpoint_artifact
from model_diffing.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseAcausalTrainer(BaseTrainer[TConfig, AcausalCrosscoder[TAct]]):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseActivationsDataloader,
        crosscoder: AcausalCrosscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
        crosscoding_dims: CrosscodingDims,
    ):
        super().__init__(cfg, activations_dataloader, crosscoder, wandb_run, device, save_dir)

        self.crosscoding_dims = crosscoding_dims

    def run_batch(self, batch_BXD: torch.Tensor, log: bool) -> tuple[torch.Tensor, dict[str, float] | None]:
        train_res = self.crosscoder.forward_train(batch_BXD)
        self.firing_tracker.add_batch(train_res.latents_BL)
        return self._calculate_loss_and_log(batch_BXD, train_res, log=log)

    def _maybe_save_model(self, scaling_factors_X: torch.Tensor) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            self.crosscoder.with_folded_scaling_factors(scaling_factors_X, scaling_factors_X).save(checkpoint_path)

            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        batch_BXD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...


    def _step_logs(self) -> dict[str, Any]:
        log_dict = super()._step_logs()
        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.tokens_since_fired_L)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})

            if (model_dim := self.crosscoding_dims.get("model")) is not None and len(model_dim) == 2:
                W_dec_LXoDo = self.crosscoder.W_dec_LXD.detach().cpu()
                log_dict.update(create_cosine_sim_and_relative_norm_histograms(W_dec_LXoDo, model_dim.index_labels))


        return log_dict

    def _get_fvu_dict(self, batch_BMD: torch.Tensor, recon_acts_BMD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            batch_BMD,
            recon_acts_BMD,
            *((dim.name, dim.index_labels) for dim in self.crosscoding_dims.values()),
        )
