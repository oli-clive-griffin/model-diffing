from abc import abstractmethod
from itertools import islice
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import torch as t
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.log import logger
from model_diffing.models import InitStrategy
from model_diffing.models.acausal_crosscoder.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.scripts.base_trainer import validate_num_steps_per_epoch
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import (
    build_lr_scheduler,
    build_optimizer,
    create_cosine_sim_and_relative_norm_histograms_diffing,
    wandb_histogram,
)
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact
from model_diffing.utils import not_none

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class IdenticalLatentsInit(InitStrategy[AcausalCrosscoder[Any]]):
    """
    Init strategy that first applies a regular init, and then sets the decoder weight such that each model
    has the same shared decoder weights for the first n_shared_latents.
    """

    def __init__(
        self,
        first_init: InitStrategy[AcausalCrosscoder[Any]],
        n_shared_latents: int,
    ):
        self.first_init = first_init
        self.n_shared_latents = n_shared_latents

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[Any]) -> None:
        assert cc.W_dec_HXD.shape[1] == 2, "expected the model dimension to be 2"

        # do the regular init
        self.first_init.init_weights(cc)

        # BUT: sync the shared decoder weights
        cc.W_dec_HXD[: self.n_shared_latents, 0].copy_(cc.W_dec_HXD[: self.n_shared_latents, 1])

        assert (cc.W_dec_HXD[: self.n_shared_latents, 0] == cc.W_dec_HXD[: self.n_shared_latents, 1]).all()


class BaseDiffingTrainer(Generic[TConfig, TAct]):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseModelHookpointActivationsDataloader,
        crosscoder: AcausalCrosscoder[TAct],
        model_dim_cc_idx: int,
        n_shared_latents: int,
        wandb_run: Run,
        device: torch.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        self.cfg = cfg
        self.model_dim_cc_idx = model_dim_cc_idx
        assert crosscoder.crosscoding_dims[model_dim_cc_idx] == 2, "expected the model dimension to be 2"
        self.n_shared_latents = n_shared_latents
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

    def train(self) -> None:
        scaling_factors_M = self.activations_dataloader.get_norm_scaling_factors_MP()[:, 0].to(self.device)
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
                    self.crosscoder.with_folded_scaling_factors(scaling_factors_M).save(checkpoint_path)

                    if self.cfg.upload_saves_to_wandb and not self.wandb_run.disabled:
                        artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                        self.wandb_run.log_artifact(artifact)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BMD.shape[0]

                self.step += 1
            self.epoch += 1

        self.wandb_run.finish()

    def _train_step(self, batch_BMD: t.Tensor) -> None:
        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self.optimizer.zero_grad()

        train_res = self.crosscoder.forward_train(batch_BMD)
        self.firing_tracker.add_batch(train_res.hidden_BH)

        loss = self._calculate_loss_and_log(batch_BMD, train_res)

        loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()

        if self.step % self.cfg.gradient_accumulation_steps_per_batch == 0:
            self._synchronise_shared_weight_grads()
            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self.optimizer.step()
            self._lr_step()

    @abstractmethod
    def _calculate_loss_and_log(self, batch_BMD: t.Tensor, train_res: AcausalCrosscoder.ForwardResult) -> t.Tensor: ...

    def _synchronise_shared_weight_grads(self) -> None:
        assert self.crosscoder.W_dec_HXD.shape[1 + self.model_dim_cc_idx] == 2, "expected the model dimension to be 2"
        assert self.crosscoder.W_dec_HXD.grad is not None
        model_0_grad = self.crosscoder.W_dec_HXD.grad[: self.n_shared_latents, 0]
        model_1_grad = self.crosscoder.W_dec_HXD.grad[: self.n_shared_latents, 1]
        assert model_0_grad is not None and model_1_grad is not None

        summed_grad = model_0_grad + model_1_grad
        model_0_grad.copy_(summed_grad)
        model_1_grad.copy_(summed_grad)
        assert (
            not_none(self.crosscoder.W_dec_HXD.grad[: self.n_shared_latents, 0])
            == not_none(self.crosscoder.W_dec_HXD.grad[: self.n_shared_latents, 1])
        ).all()

        assert (
            self.crosscoder.W_dec_HXD[: self.n_shared_latents, 0]
            == self.crosscoder.W_dec_HXD[: self.n_shared_latents, 1]
        ).all()

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def _common_logs(self) -> dict[str, Any]:
        logs = {
            "train/epoch": self.epoch,
            "train/unique_tokens_trained": self.unique_tokens_trained,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if self.step % (self.cfg.log_every_n_steps * 10) == 0:  # type: ignore
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.examples_since_fired_A)
            logs.update({"media/tokens_since_fired": tokens_since_fired_hist})

            W_dec_HiMD = self.crosscoder.W_dec_HXD[self.n_shared_latents :].detach()
            logs.update(create_cosine_sim_and_relative_norm_histograms_diffing(W_dec_HMD=W_dec_HiMD))

        return logs
