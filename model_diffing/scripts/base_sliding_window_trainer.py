from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch as t
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.token_hookpoint_dataloader import BaseTokenHookpointActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.scripts.base_trainer import TConfig, validate_num_steps_per_epoch
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer, dict_join, wandb_histogram
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact
from model_diffing.utils import get_fvu_dict

TAct = TypeVar("TAct", bound=ActivationFunction)


class BiTokenCCWrapper(nn.Module, Generic[TAct]):
    def __init__(
        self,
        single_token_cc: AcausalCrosscoder[TAct],
        double_token_cc: AcausalCrosscoder[TAct],
    ):
        super().__init__()

        assert single_token_cc.crosscoding_dims[0] == 1  # token
        assert len(single_token_cc.crosscoding_dims) == 2  # (token, hookpoint)
        self.single_cc = single_token_cc

        assert double_token_cc.crosscoding_dims[0] == 2  # token
        assert len(double_token_cc.crosscoding_dims) == 2  # (token, hookpoint)
        self.double_cc = double_token_cc

    @dataclass
    class TrainResult:
        recon_single1_B1PD: t.Tensor
        recon_single2_B1PD: t.Tensor
        recon_double_B2PD: t.Tensor
        hidden_single1_BL: t.Tensor
        hidden_single2_BL: t.Tensor
        hidden_double_BL: t.Tensor

    def forward_train(self, x_BTPD: t.Tensor) -> TrainResult:
        assert x_BTPD.shape[1] == 2

        output_single1 = self.single_cc.forward_train(x_BTPD[:, 0][:, None])
        output_single2 = self.single_cc.forward_train(x_BTPD[:, 1][:, None])
        output_both = self.double_cc.forward_train(x_BTPD)

        return self.TrainResult(
            recon_single1_B1PD=output_single1.recon_acts_BXD,
            recon_single2_B1PD=output_single2.recon_acts_BXD,
            recon_double_B2PD=output_both.recon_acts_BXD,
            hidden_single1_BL=output_single1.latents_BL,
            hidden_single2_BL=output_single2.latents_BL,
            hidden_double_BL=output_both.latents_BL,
        )

    # stub forward for appeasing the nn.Module interface, but we don't use it
    def forward(self, x_BTPD: t.Tensor) -> t.Tensor:
        raise NotImplementedError("This method should not be called")


class BaseSlidingWindowCrosscoderTrainer(Generic[TAct, TConfig], ABC):
    LOG_HISTOGRAMS_EVERY_N_LOGS = 10

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseTokenHookpointActivationsDataloader,
        crosscoders: BiTokenCCWrapper[TAct],
        wandb_run: Run,
        device: t.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.hookpoints = hookpoints

        self.crosscoders = crosscoders

        self.optimizer = build_optimizer(cfg.optimizer, self.crosscoders.parameters())

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader.num_batches()
        )

        self.total_steps = self.num_steps_per_epoch * (cfg.epochs or 1)
        logger.info(
            f"Total steps: {self.total_steps} (num_steps_per_epoch: {self.num_steps_per_epoch}, epochs: {cfg.epochs})"
        )

        self.lr_scheduler = (
            build_lr_scheduler(cfg.optimizer, self.total_steps) if cfg.optimizer.type == "adam" else None
        )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.firing_tracker = FiringTracker(
            activation_size=self.crosscoders.single_cc.n_latents
            + self.crosscoders.double_cc.n_latents
            + self.crosscoders.single_cc.n_latents,
            device=self.device,
        )

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        scaling_factors_TP = self.activations_dataloader.get_norm_scaling_factors_TP().to(self.device)
        scaling_factor_1P = scaling_factors_TP.mean(dim=0, keepdim=True)

        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            self._do_epoch(scaling_factors_TP, scaling_factor_1P)
            self.epoch += 1

    def _do_epoch(self, scaling_factors_TP: t.Tensor, scaling_factor_1P: t.Tensor) -> None:
        epoch_dataloader_BTPD = self.activations_dataloader.get_activations_iterator_BTPD()

        for _ in tqdm(
            range(self.num_steps_per_epoch),
            desc="Epoch Train Steps",
            total=self.num_steps_per_epoch,
            smoothing=0.15,  # this loop is bursty because of activation harvesting
        ):
            self.optimizer.zero_grad()

            log_dicts: list[dict[str, float]] = []
            log = self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0

            for _ in range(self.cfg.gradient_accumulation_steps_per_batch):
                batch_BTPD = next(epoch_dataloader_BTPD)
                batch_BTPD = batch_BTPD.to(self.device)

                res = self.crosscoders.forward_train(batch_BTPD)
                hidden_B3l = t.cat([res.hidden_single1_BL, res.hidden_double_BL, res.hidden_single2_BL], dim=-1)
                self.firing_tracker.add_batch(hidden_B3l)

                loss, log_dict = self._calculate_loss_and_log(batch_BTPD, res, log=log)

                loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()
                if log_dict is not None:
                    log_dicts.append(log_dict)

            if log_dicts:
                batch_log_dict_avgs = {
                    **{k: sum(v) / len(v) for k, v in dict_join(log_dicts).items()},
                    **self._step_logs(),
                }
                self.wandb_run.log(batch_log_dict_avgs)

            self._maybe_save_model(scaling_factors_TP, scaling_factor_1P)

            clip_grad_norm_(self.crosscoders.parameters(), 1.0)
            self._lr_step()
            self.optimizer.step()
            if self.epoch == 0:
                self.unique_tokens_trained += batch_BTPD.shape[0]
            self.step += 1

    def _maybe_save_model(self, scaling_factors_TP: t.Tensor, scaling_factor_1P: t.Tensor) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            step_dir_single = self.save_dir / f"epoch_{self.epoch}_step_{self.step}_single"
            step_dir_double = self.save_dir / f"epoch_{self.epoch}_step_{self.step}_double"

            self.crosscoders.single_cc.with_folded_scaling_factors(scaling_factor_1P).save(step_dir_single)
            self.crosscoders.double_cc.with_folded_scaling_factors(scaling_factors_TP).save(step_dir_double)

            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(step_dir_single, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

                artifact = create_checkpoint_artifact(step_dir_double, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        batch_BTPD: t.Tensor,
        res: BiTokenCCWrapper.TrainResult,
        log: bool,
    ) -> tuple[t.Tensor, dict[str, float] | None]: ...

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

            # log bias histograms?

        return log_dict

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def _get_fvu_dict(self, batch_BTPD: t.Tensor, reconstructed_acts_BTPD: t.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            batch_BTPD,
            reconstructed_acts_BTPD,
            ("token", [0, 1]),
            ("hookpoint", self.hookpoints),
        )
