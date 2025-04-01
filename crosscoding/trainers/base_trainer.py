import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from crosscoding.data.activations_dataloader import ActivationsDataloader
from crosscoding.log import logger
from crosscoding.models.base_crosscoder import BaseCrosscoder
from crosscoding.trainers.config_common import BaseExperimentConfig, BaseTrainConfig
from crosscoding.trainers.firing_tracker import FiringTracker
from crosscoding.trainers.utils import (
    build_lr_scheduler,
    build_optimizer,
    dict_join,
    wandb_histogram,
)

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TCC = TypeVar("TCC", bound=BaseCrosscoder[Any])
TBatch = TypeVar("TBatch")


class BaseTrainer(Generic[TConfig, TCC, TBatch], ABC):
    LOG_HISTOGRAMS_EVERY_N_LOGS = 10

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: ActivationsDataloader[TBatch],
        crosscoder: TCC,
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader

        self.crosscoder = crosscoder
        self.wandb_run = wandb_run
        self.device = device

        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, cfg.num_steps) if cfg.optimizer.type == "adam" else None

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.firing_tracker = FiringTracker(activation_size=crosscoder.n_latents, device=self.device)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self) -> None:
        # scaling_factors_MP = self.activations_dataloader.get_norm_scaling_factors_MP().to(self.device)
        epoch_dataloader = self.activations_dataloader.get_activations_iterator()

        for _ in tqdm(
            range(self.cfg.num_steps),
            desc="Train Steps",
            smoothing=0.15,  # this loop is bursty because of activation harvesting
        ):
            self._lr_step()
            self.optimizer.zero_grad()

            log_dicts: list[dict[str, float]] = []
            log = self.step % self.cfg.log_every_n_steps == 0

            for _ in range(self.cfg.gradient_accumulation_steps_per_batch):
                loss, log_dict, tokens_trained = self.run_batch(next(epoch_dataloader), log)
                if self.epoch == 0:
                    self.unique_tokens_trained += tokens_trained

                loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()
                if log_dict is not None:
                    log_dicts.append(log_dict)

            self._after_forward_passes()

            if log_dicts:
                batch_log_dict_avgs = {
                    **{k: sum(v) / len(v) for k, v in dict_join(log_dicts).items()},
                    **self._step_logs(),
                }
                self.wandb_run.log(batch_log_dict_avgs, step=self.step)

            self._maybe_save_model()

            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()
            self.step += 1

        self.wandb_run.finish()

    def _after_forward_passes(self): ...

    @abstractmethod
    def run_batch(self, batch: TBatch, log: bool) -> tuple[torch.Tensor, dict[str, float] | None, int]: ...

    @abstractmethod
    def _maybe_save_model(self) -> None: ...

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

        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.tokens_since_fired_L)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})
            if self.crosscoder.b_enc_L is not None:
                log_dict["b_enc"] = wandb_histogram(self.crosscoder.b_enc_L)

        return log_dict


def save_config(config: BaseExperimentConfig) -> None:
    config.save_dir.mkdir(parents=True, exist_ok=True)
    with open(config.save_dir / "experiment_config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f)
    logger.info(f"Saved config to {config.save_dir / 'experiment_config.yaml'}")


TCfg = TypeVar("TCfg", bound=BaseExperimentConfig)


def run_exp(build_trainer: Callable[[TCfg], Any], cfg_cls: type[TCfg]) -> Callable[[Path], None]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def inner(config_path: Path) -> None:
        config_path = Path(config_path)
        assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
        assert Path(config_path).exists(), f"Config file {config_path} does not exist."
        logger.info("Loading config...")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Loaded config (raw):\n{config_dict}")
        config = cfg_cls(**config_dict)
        logger.info(f"Loaded config (parsed):\n{config.model_dump_json(indent=2)}")
        config.experiment_name += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        logger.info(f"over-wrote experiment_name: {config.experiment_name}")
        logger.info(f"saving in save_dir: {config.save_dir}")
        save_config(config)
        logger.info("Building trainer")
        trainer = build_trainer(config)
        logger.info("Training")
        trainer.train()

    return inner
