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
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.models.diffing_crosscoder import DiffingCrosscoder
from model_diffing.scripts.base_trainer import validate_num_steps_per_epoch
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import (
    build_lr_scheduler,
    build_optimizer,
    wandb_histogram,
)

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseDiffingTrainer(Generic[TConfig, TAct]):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseModelHookpointActivationsDataloader,
        crosscoder: DiffingCrosscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        self.cfg = cfg
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

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def train(self) -> None:
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

                # if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                #     checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

                # with self.crosscoder.temporarily_fold_activation_scaling(
                #     self.activations_dataloader.get_norm_scaling_factors_MP().to(self.device)
                # ):
                #     save_model(self.crosscoder, checkpoint_path)

                # artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                # self.wandb_run.log_artifact(artifact)

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

            # if self.n_models == 2:
            #     W_dec_HXD = self.crosscoder.W_dec_HXD.detach().cpu()
            #     assert W_dec_HXD.shape[1:-1] == (self.n_models, self.n_hookpoints)
            #     logs.update(create_cosine_sim_and_relative_norm_histograms(W_dec_HXD, self.hookpoints))

        return logs

    @abstractmethod
    def _train_step(self, batch_BMD: t.Tensor) -> None: ...
