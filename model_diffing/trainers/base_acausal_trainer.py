from abc import abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from model_diffing.data.model_hookpoint_dataloader import (
    BaseModelHookpointActivationsDataloader,
)
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.trainers.base_trainer import BaseTrainer
from model_diffing.trainers.config_common import BaseTrainConfig
from model_diffing.trainers.utils import (
    create_cosine_sim_and_relative_norm_histograms,
    wandb_histogram,
)
from model_diffing.trainers.wandb_utils.main import create_checkpoint_artifact
from model_diffing.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)

class BaseModelHookpointAcausalTrainer(BaseTrainer[TConfig, AcausalCrosscoder[TAct]]):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseModelHookpointActivationsDataloader,
        crosscoder: AcausalCrosscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        super().__init__(cfg, activations_dataloader, crosscoder, wandb_run, device, hookpoints, save_dir)
        assert len(self.crosscoder.crosscoding_dims) == 2
        self.n_models = self.crosscoder.crosscoding_dims[0]
        self.n_hookpoints = self.crosscoder.crosscoding_dims[1]

    def run_batch(self, batch_BMPD: torch.Tensor, log: bool) -> tuple[torch.Tensor, dict[str, float] | None]:
        train_res = self.crosscoder.forward_train(batch_BMPD)
        self.firing_tracker.add_batch(train_res.latents_BL)
        return self._calculate_loss_and_log(batch_BMPD, train_res, log=log)

    def _maybe_save_model(self, scaling_factors_MP: torch.Tensor) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            self.crosscoder.with_folded_scaling_factors(scaling_factors_MP, scaling_factors_MP).save(checkpoint_path)

            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def _step_logs(self) -> dict[str, Any]:
        log_dict: dict[str, Any] = {
            "train/epoch": self.epoch,
            "train/unique_tokens_trained": self.unique_tokens_trained,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if (
            self.cfg.log_every_n_steps is not None
            and self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0
        ):
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.tokens_since_fired_L)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})

            if self.crosscoder.n_models == 2:
                W_dec_LXoDo = self.crosscoder._W_dec_LXoDo.detach().cpu()
                assert W_dec_LXoDo.shape[1:-1] == (self.n_models, self.n_hookpoints)
                log_dict.update(create_cosine_sim_and_relative_norm_histograms(W_dec_LXoDo, self.hookpoints))

            if self.crosscoder.b_enc_L is not None:
                log_dict["b_enc_values"] = wandb_histogram(self.crosscoder.b_enc_L)

            if self.crosscoder.b_dec_XD is not None:
                for i in range(self.n_models):
                    for j in range(self.n_hookpoints):
                        log_dict[f"b_dec_values_m{i}_hp{j}"] = wandb_histogram(self.crosscoder.b_dec_XD[i, j])

        return log_dict

    def _get_fvu_dict(self, batch_BMPD: torch.Tensor, recon_acts_BMPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            batch_BMPD,
            recon_acts_BMPD,
            ("model", list(range(self.n_models))),
            ("hookpoint", self.hookpoints),
        )