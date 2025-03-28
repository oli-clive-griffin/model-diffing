from abc import abstractmethod
from typing import Any, TypeVar

import torch

from model_diffing.models.crosscoder import CrossLayerTranscoder, _BaseCrosscoder
from model_diffing.scripts.base_acausal_trainer import BaseTrainer
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import (
    wandb_histogram,
)
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact
from model_diffing.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TCC = TypeVar("TCC", bound=_BaseCrosscoder[Any])


class BaseCrossLayerTranscoderTrainer(BaseTrainer[TConfig, CrossLayerTranscoder[Any]]):
    def run_batch(self, batch_BMPD: torch.Tensor, log: bool) -> tuple[torch.Tensor, dict[str, float] | None]:
        assert batch_BMPD.shape[1] == 1, "we must have one model"
        batch_BPD = batch_BMPD[:, 0]
        in_BD, out_BPD = batch_BPD[:, 0], batch_BPD[:, 1:]

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

    def _step_logs(self) -> dict[str, Any]:
        logs = super()._step_logs()
        if (
            self.cfg.log_every_n_steps is not None
            and self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0
            and self.crosscoder.b_dec_PD is not None
        ):
            for i, hp_name in enumerate(self.hookpoints[1:]):
                logs[f"b_dec_{hp_name}"] = wandb_histogram(self.crosscoder.b_dec_PD[i])
        return logs

    def _get_fvu_dict(self, y_BPD: torch.Tensor, recon_y_BPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            y_BPD,
            recon_y_BPD,
            ("hookpoint", self.hookpoints),
        )

    def _maybe_save_model(self, scaling_factors_MP: torch.Tensor) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            self.crosscoder.with_folded_scaling_factors(scaling_factors_MP[:, 0], scaling_factors_MP[:, 1:]).save(
                checkpoint_path
            )
            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)