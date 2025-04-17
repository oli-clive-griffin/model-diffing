import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch, ModelHookpointActivationsDataloader
from crosscode.log import logger
from crosscode.trainers.config_common import BaseExperimentConfig
from crosscode.trainers.utils import dict_mean
from crosscode.trainers.wandb_utils.main import create_checkpoint_artifact


def save_config(config: BaseExperimentConfig) -> None:
    config.save_dir.mkdir(parents=True, exist_ok=True)
    with open(config.save_dir / "experiment_config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f)
    logger.info(f"Saved config to {config.save_dir / 'experiment_config.yaml'}")


class ModelWrapper:
    @abstractmethod
    def run_batch(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def before_backward_pass(self) -> None:
        pass

    @abstractmethod
    def save(self, step: int) -> Path: ...

    @abstractmethod
    def expensive_logs(self) -> dict[str, Any]: ...


class Trainer:
    def __init__(
        self,
        num_steps: int,
        gradient_accumulation_steps_per_batch: int,
        log_every_n_steps: int,
        save_every_n_steps: int | None,
        upload_saves_to_wandb: bool,
        activations_dataloader: ModelHookpointActivationsDataloader,
        model_wrapper: ModelWrapper,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        wandb_run: Run,
    ):
        self.num_steps = num_steps
        self.gradient_accumulation_steps_per_batch = gradient_accumulation_steps_per_batch
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        self.upload_saves_to_wandb = upload_saves_to_wandb
        self.activations_dataloader = activations_dataloader
        self.model_wrapper = model_wrapper
        self.wandb_run = wandb_run
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = 0
        self.unique_tokens_trained = 0

    def train(self) -> None:
        dataloader = self.activations_dataloader.get_activations_iterator()
        for _ in tqdm(
            range(self.num_steps),
            desc="Train Steps",
            smoothing=0.15,  # this loop is bursty because of activation harvesting
        ):
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            log_dicts: list[dict[str, float]] = []
            log = self.step % self.log_every_n_steps == 0
            for _ in range(self.gradient_accumulation_steps_per_batch):
                batch = next(dataloader)
                loss, log_dict = self.model_wrapper.run_batch(self.step, batch, log)
                self.unique_tokens_trained += batch.activations_BMPD.shape[0]
                loss.div(self.gradient_accumulation_steps_per_batch).backward()
                if log_dict is not None:
                    log_dicts.append(log_dict)
            if log_dicts:
                batch_log_dict_avgs = {
                    "train/unique_tokens_trained": self.unique_tokens_trained,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    **dict_mean(log_dicts),  # take the mean of values of each key
                }
                if self.step % 100 == 0:
                    batch_log_dict_avgs.update(self.model_wrapper.expensive_logs(self.step))
                self.wandb_run.log(batch_log_dict_avgs, step=self.step)
            if self.save_every_n_steps is not None and self.step % self.save_every_n_steps == 0:
                dir = self.model_wrapper.save(self.step)
                if self.upload_saves_to_wandb:
                    artifact = create_checkpoint_artifact(dir, self.wandb_run.id, self.step)
                    self.wandb_run.log_artifact(artifact)
            self.model_wrapper.before_backward_pass()
            self.optimizer.step()
            self.step += 1
        self.wandb_run.finish()


TExperimentConfig = TypeVar("TExperimentConfig", bound=BaseExperimentConfig)


def run_exp(
    build_trainer: Callable[[TExperimentConfig], Any], cfg_cls: type[TExperimentConfig]
) -> Callable[[Path], None]:
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
