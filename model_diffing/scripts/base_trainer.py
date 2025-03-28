import os
from abc import abstractmethod
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import yaml  # type: ignore
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.model_hookpoint_dataloader import (
    BaseModelHookpointActivationsDataloader,
)
from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import (
    build_lr_scheduler,
    build_optimizer,
    create_cosine_sim_and_relative_norm_histograms,
    dict_join,
    wandb_histogram,
)
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact
from model_diffing.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseModelHookpointTrainer(Generic[TConfig, TAct]):
    LOG_HISTOGRAMS_EVERY_N_LOGS = 10

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

        self.lr_scheduler = (
            build_lr_scheduler(cfg.optimizer, self.total_steps) if cfg.optimizer.type == "adam" else None
        )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.firing_tracker = FiringTracker(activation_size=crosscoder.n_latents, device=self.device)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self) -> None:
        scaling_factors_MP = self.activations_dataloader.get_norm_scaling_factors_MP().to(self.device)
        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            epoch_dataloader_BMPD = self.activations_dataloader.get_activations_iterator_BMPD()
            self._do_epoch(scaling_factors_MP, epoch_dataloader_BMPD)
            self.epoch += 1

        self.wandb_run.finish()

    def _do_epoch(
        self,
        scaling_factors_MP: torch.Tensor,
        epoch_dataloader_BMPD: Iterator[torch.Tensor],
    ) -> None:
        for _ in tqdm(
            range(self.num_steps_per_epoch),
            desc="Epoch Train Steps",
            total=self.num_steps_per_epoch,
            smoothing=0.15,  # this loop is bursty because of activation harvesting
        ):
            self._lr_step()
            self.optimizer.zero_grad()

            log_dicts: list[dict[str, float]] = []
            log = self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0

            for _ in range(self.cfg.gradient_accumulation_steps_per_batch):
                batch_BMPD = next(epoch_dataloader_BMPD).to(self.device)
                train_res = self.crosscoder.forward_train(batch_BMPD)
                self.firing_tracker.add_batch(train_res.latents_BL)

                loss, log_dict = self._calculate_loss_and_log(batch_BMPD, train_res, log=log)

                loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()
                if log_dict is not None:
                    log_dicts.append(log_dict)

            if log_dicts:
                batch_log_dict_avgs = {
                    **{k: sum(v) / len(v) for k, v in dict_join(log_dicts).items()},
                    **self._step_logs(),
                }
                self.wandb_run.log(batch_log_dict_avgs, step=self.step)

            self._maybe_save_model(scaling_factors_MP)

            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()
            if self.epoch == 0:
                self.unique_tokens_trained += batch_BMPD.shape[0]
            self.step += 1

    def _maybe_save_model(self, scaling_factors_MP: torch.Tensor) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            self.crosscoder.with_folded_scaling_factors(scaling_factors_MP).save(checkpoint_path)

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

            if self.n_models == 2:
                W_dec_LXD = self.crosscoder.W_dec_LXD.detach().cpu()
                assert W_dec_LXD.shape[1:-1] == (self.n_models, self.n_hookpoints)
                log_dict.update(create_cosine_sim_and_relative_norm_histograms(W_dec_LXD, self.hookpoints))

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
