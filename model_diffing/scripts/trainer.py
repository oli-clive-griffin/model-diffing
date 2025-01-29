from abc import abstractmethod
from itertools import islice
from pathlib import Path

import torch
import wandb
from einops import rearrange
from wandb.sdk.wandb_run import Run

from model_diffing.analysis.visualization import create_visualizations
from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer, estimate_norm_scaling_factor_ML
from model_diffing.utils import save_model_and_config


class BaseTrainer[TConfig: BaseTrainConfig]:
    step: int
    epoch: int
    unique_tokens_trained: int

    # I've tried to make this invariant as obvious as possible in this type signature:
    # If training without epochs (epochs=1), we need to provide num_steps
    # However, if training with epochs, we don't need to limit the number of steps per epoch,
    # just loop through the dataloader
    # epochs_steps: tuple[Literal[1], int] | tuple[int, None]

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseActivationsDataloader,
        crosscoder: AcausalCrosscoder,
        wandb_run: Run | None,
        device: torch.device,
        layers_to_harvest: list[int],
        experiment_name: str,
    ):
        self.cfg = cfg
        self.crosscoder = crosscoder
        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())

        self.epochs = cfg.epochs

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader
        )

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.num_steps_per_epoch)

        self.activations_dataloader = activations_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.save_dir = Path(cfg.base_save_dir) / experiment_name if cfg.base_save_dir is not None else None

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        logger.info("Estimating norm scaling factors (model, layer)")

        norm_scaling_factors_ML = estimate_norm_scaling_factor_ML(
            self.activations_dataloader.get_shuffled_activations_iterator_BMLD(),
            self.device,
            self.cfg.n_batches_for_norm_estimate,
        )

        norm_scaling_factors_ML1 = rearrange(norm_scaling_factors_ML, "m l -> m l 1")

        logger.info(f"Norm scaling factors (model, layer): {norm_scaling_factors_ML}")

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )

        for _ in range(self.epochs or 1):
            epoch_dataloader = islice(
                self.activations_dataloader.get_shuffled_activations_iterator_BMLD(),
                self.num_steps_per_epoch,
            )

            for example_BMLD in epoch_dataloader:
                batch_BMLD = example_BMLD.to(self.device)
                batch_BMLD = batch_BMLD * norm_scaling_factors_ML1

                log_dict = {
                    **self._train_step(batch_BMLD),
                    "train/step": self.step,
                    "train/epoch": self.epoch,
                    "train/unique_tokens_trained": self.unique_tokens_trained,
                }

                if self.wandb_run:
                    if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
                        self.wandb_run.log(log_dict, step=self.step)

                    if (
                        self.cfg.log_visualizations_every_n_steps is not None
                        and self.step % self.cfg.log_visualizations_every_n_steps == 0
                    ):
                        visualizations = create_visualizations(
                            self.crosscoder.W_dec_HMLD.detach().cpu(), self.layers_to_harvest
                        )
                        self.wandb_run.log(
                            {f"visualizations/{k}": wandb.Plotly(v) for k, v in visualizations.items()},
                            step=self.step,
                        )

                if (
                    self.save_dir is not None
                    and self.cfg.save_every_n_steps is not None
                    and self.step % self.cfg.save_every_n_steps == 0
                ):
                    save_model_and_config(
                        config=self.cfg,
                        save_dir=self.save_dir,
                        model=self.crosscoder,
                        epoch=self.step,
                    )

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BMLD.shape[0]

                self.step += 1
            self.epoch += 1

    @abstractmethod
    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]: ...


def validate_num_steps_per_epoch(
    epochs: int | None,
    num_steps_per_epoch: int | None,
    num_steps: int | None,
    activations_dataloader: BaseActivationsDataloader,
) -> int:
    if epochs is not None:
        if num_steps is not None:
            raise ValueError("num_steps must not be provided if using epochs")

        dataloader_num_batches = activations_dataloader.num_batches()
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
