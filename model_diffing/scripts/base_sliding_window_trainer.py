from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
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
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact

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
        hidden_single1_BH: t.Tensor
        hidden_single2_BH: t.Tensor
        hidden_double_BH: t.Tensor

    def forward_train(self, x_BTPD: t.Tensor) -> TrainResult:
        assert x_BTPD.shape[1] == 2

        output_single1 = self.single_cc.forward_train(x_BTPD[:, 0][:, None])
        output_single2 = self.single_cc.forward_train(x_BTPD[:, 1][:, None])
        output_both = self.double_cc.forward_train(x_BTPD)

        return self.TrainResult(
            recon_single1_B1PD=output_single1.recon_acts_BXD,
            recon_single2_B1PD=output_single2.recon_acts_BXD,
            recon_double_B2PD=output_both.recon_acts_BXD,
            hidden_single1_BH=output_single1.hidden_BH,
            hidden_single2_BH=output_single2.hidden_BH,
            hidden_double_BH=output_both.hidden_BH,
        )

    # stub forward for appeasing the nn.Module interface, but we don't use it
    def forward(self, x_BTPD: t.Tensor) -> t.Tensor:
        raise NotImplementedError("This method should not be called")


class BaseSlidingWindowCrosscoderTrainer(Generic[TAct, TConfig], ABC):
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
            activation_size=self.crosscoders.single_cc.hidden_dim
            + self.crosscoders.double_cc.hidden_dim
            + self.crosscoders.single_cc.hidden_dim
        )

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        scaling_factors_TP = self.activations_dataloader.get_norm_scaling_factors_TP().to(self.device)
        scaling_factor_1P = scaling_factors_TP.mean(dim=0, keepdim=True)

        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            for batch_BTPD in tqdm(
                islice(self.activations_dataloader.get_activations_iterator_BTPD(), self.num_steps_per_epoch),
                desc="Epoch Train Steps",
                total=self.num_steps_per_epoch,
                smoothing=0.15,  # this loop is bursty because of activation harvesting
            ):
                batch_BTPD = batch_BTPD.to(self.device)

                self._train_step(batch_BTPD)

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

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BTPD.shape[0]

                self.step += 1
            self.epoch += 1

    def _train_step(self, batch_BTPD: t.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        res = self.crosscoders.forward_train(batch_BTPD)
        hidden_B3h = t.cat([res.hidden_single1_BH, res.hidden_double_BH, res.hidden_single2_BH], dim=-1)
        self.firing_tracker.add_batch(hidden_B3h)

        loss = self._calculate_loss_and_log(batch_BTPD, res)

        loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()

        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            clip_grad_norm_(self.crosscoders.parameters(), 1.0)
            self.optimizer.step()
            self._lr_step()

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        batch_BTPD: t.Tensor,
        res: BiTokenCCWrapper.TrainResult,
    ) -> t.Tensor: ...

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def _common_logs(self) -> dict[str, Any]:
        return {
            "train/epoch": self.epoch,
            "train/unique_tokens_trained": self.unique_tokens_trained,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }
