import os
from abc import abstractmethod
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import wandb
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer, wandb_histogram
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact
from model_diffing.utils import SaveableModule

# using python3.11 generics because it's better supported by GPU providers
TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=SaveableModule)


class BaseModelHookpointTrainer(Generic[TConfig, TAct]):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseModelHookpointActivationsDataloader,
        crosscoder: AcausalCrosscoder[TAct],
        wandb_run: Run | None,
        device: torch.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader

        assert len(crosscoder.crosscoding_dims) == 2, (
            "crosscoder must have 2 crosscoding dimensions (model, hookpoint). (They can be singleton dimensions)"
        )
        self.n_models, self.n_hookpoints = crosscoder.crosscoding_dims

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

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.total_steps)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # buffer to track when features activation, used for logging dead features
        self.firing_tracker = FiringTracker(
            activation_size=crosscoder.hidden_dim,
            # length=1_000_000,  # TODO(oli): make this dynamic
            # device=self.device,
        )

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def get_firing_percentage_hist(self) -> wandb.Histogram:
        return wandb_histogram(self.firing_tracker.steps_since_fired_A)

    def train(self) -> None:
        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            epoch_dataloader_BMPD = self.activations_dataloader.get_shuffled_activations_iterator_BMPD()
            epoch_dataloader_BMPD = islice(epoch_dataloader_BMPD, self.num_steps_per_epoch)

            for batch_BMPD in tqdm(epoch_dataloader_BMPD, desc="Train Steps"):
                batch_BMPD = batch_BMPD.to(self.device)

                self._train_step(batch_BMPD)

                # TODO(oli): get wandb checkpoint saving working

                if self.cfg.save_every_n_steps is not None and (self.step + 1) % self.cfg.save_every_n_steps == 0:
                    checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

                    with self.crosscoder.temporarily_fold_activation_scaling(
                        self.activations_dataloader.get_norm_scaling_factors_MP()
                    ):
                        save_model(self.crosscoder, checkpoint_path)

                    if self.wandb_run is not None:
                        artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id)
                        self.wandb_run.log_artifact(artifact)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BMPD.shape[0]

                self.step += 1
            self.epoch += 1

    @abstractmethod
    def _train_step(self, batch_BMPD: torch.Tensor) -> None: ...


def validate_num_steps_per_epoch(
    epochs: int | None,
    num_steps_per_epoch: int | None,
    num_steps: int | None,
    dataloader_num_batches: int | None,
) -> int:
    if epochs is not None:
        if num_steps is not None:
            raise ValueError("num_steps must not be provided if using epochs")

        if dataloader_num_batches is None:
            raise ValueError(
                "activations_dataloader must have a length if using epochs, "
                "as we need to know how to schedule the learning rate"
            )

        if num_steps_per_epoch is None:
            return dataloader_num_batches
        else:
            if dataloader_num_batches < num_steps_per_epoch:
                logger.warning(
                    f"num_steps_per_epoch ({num_steps_per_epoch}) is greater than the number "
                    f"of batches in the dataloader ({dataloader_num_batches}), so we will only "
                    "train for the number of batches in the dataloader"
                )
                return dataloader_num_batches
            else:
                return num_steps_per_epoch

    # not using epochs
    if num_steps is None:
        raise ValueError("num_steps must be provided if not using epochs")
    if num_steps_per_epoch is not None:
        raise ValueError("num_steps_per_epoch must not be provided if not using epochs")
    return num_steps


def save_config(config: BaseExperimentConfig) -> None:
    config.save_dir.mkdir(parents=True, exist_ok=True)
    with open(config.save_dir / "config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f)
    logger.info(f"Saved config to {config.save_dir / 'config.yaml'}")


def save_model(model: SaveableModule, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir)
    logger.info(f"Saved model to {save_dir}")


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
        config = cfg_cls(**config_dict)
        logger.info(f"Loaded config, saving in save_dir: {config.save_dir}")
        save_config(config)
        logger.info("Building trainer")
        trainer = build_trainer(config)
        logger.info("Training")
        trainer.train()

    return inner
