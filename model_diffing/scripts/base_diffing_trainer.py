from abc import abstractmethod
from itertools import islice
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import torch as t
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.scripts.base_trainer import save_model, validate_num_steps_per_epoch
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import (
    build_lr_scheduler,
    build_optimizer,
    create_cosine_sim_and_relative_norm_histograms_diffing,
    wandb_histogram,
)
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseDiffingTrainer(Generic[TConfig, TAct]):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseModelHookpointActivationsDataloader,
        crosscoder: AcausalCrosscoder[TAct],
        model_dim_cc_idx: int,
        n_shared_weights: int,
        wandb_run: Run,
        device: torch.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        self.cfg = cfg
        self.model_dim_cc_idx = model_dim_cc_idx
        assert crosscoder.crosscoding_dims[model_dim_cc_idx] == 2, "expected the model dimension to be 2"
        self.n_shared_weights = n_shared_weights
        self.activations_dataloader = activations_dataloader

        self.crosscoder = crosscoder
        self.wandb_run = wandb_run
        self.device = device
        self.hookpoints = hookpoints

        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())

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

        self.firing_tracker = FiringTracker(activation_size=crosscoder.hidden_dim)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def _synchronise_shared_weight_grads(self) -> None:
        assert self.crosscoder.W_dec_HXD.shape[1 + self.model_dim_cc_idx] == 2, "expected the model dimension to be 2"
        assert self.crosscoder.W_dec_HXD.grad is not None
        self.crosscoder.W_dec_HXD[: self.n_shared_weights, 0].grad += self.crosscoder.W_dec_HXD[
            : self.n_shared_weights, 1
        ].grad  # type: ignore
        self.crosscoder.W_dec_HXD[: self.n_shared_weights, 1].grad += self.crosscoder.W_dec_HXD[
            : self.n_shared_weights, 0
        ].grad  # type: ignore

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def train(self) -> None:
        scaling_factors_M = self.activations_dataloader.get_norm_scaling_factors_MP()[:, 0]
        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            epoch_dataloader_BMPD = self.activations_dataloader.get_activations_iterator_BMPD()
            epoch_dataloader_BMPD = islice(epoch_dataloader_BMPD, self.num_steps_per_epoch)

            batch_iter = tqdm(
                epoch_dataloader_BMPD,
                desc="Epoch Train Steps",
                total=self.num_steps_per_epoch,
                smoothing=0.2,  # this loop is bursty because of activation harvesting
            )
            for batch_BMPD in batch_iter:
                assert batch_BMPD.shape[1] == 2, "we only support 2 models for now"
                assert batch_BMPD.shape[2] == 1, "we only support 1 hookpoint for now"

                batch_BMD = batch_BMPD.squeeze(2).to(self.device)

                self._train_step(batch_BMD)

                if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                    checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"
                    with self.crosscoder.temporarily_fold_activation_scaling(scaling_factors_M.to(self.device)):
                        save_model(self.crosscoder, checkpoint_path)

                    if self.cfg.upload_saves_to_wandb:
                        artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                        self.wandb_run.log_artifact(artifact)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BMD.shape[0]

                self.step += 1
            self.epoch += 1

        self.wandb_run.finish()

    def _common_logs(self) -> dict[str, Any]:
        logs = {
            "train/epoch": self.epoch,
            "train/unique_tokens_trained": self.unique_tokens_trained,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if self.step % (self.cfg.log_every_n_steps * 10) == 0:  # type: ignore
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.examples_since_fired_A)
            logs.update({"media/tokens_since_fired": tokens_since_fired_hist})

            W_dec_HiMD = self.crosscoder.W_dec_HXD[self.n_shared_weights :].detach()
            logs.update(create_cosine_sim_and_relative_norm_histograms_diffing(W_dec_HMD=W_dec_HiMD))

        return logs

    @abstractmethod
    def _train_step(self, batch_BMD: t.Tensor) -> None: ...
