from abc import abstractmethod
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch as t
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.scripts.base_trainer import validate_num_steps_per_epoch
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.firing_tracker import FiringTracker
from model_diffing.scripts.utils import (
    build_lr_scheduler,
    build_optimizer,
    create_cosine_sim_and_relative_norm_histograms_diffing,
    dict_join,
    wandb_histogram,
)
from model_diffing.scripts.wandb_scripts.main import create_checkpoint_artifact
from model_diffing.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


# class DiffingCrosscoder(AcausalCrosscoder[TAct]):
#     CROSSCODING_DIMS = (2,)

#     def __init__(
#         self,
#         d_model: int,
#         hidden_dim: int,
#         hidden_activation: TAct,
#         skip_linear: bool = False,
#         init_strategy: InitStrategy["AcausalCrosscoder[TAct]"] | None = None,
#         dtype: t.dtype = t.float32,
#     ):
#         super().__init__(
#             crosscoding_dims=self.CROSSCODING_DIMS,
#             d_model=d_model,
#             hidden_dim=hidden_dim,
#             hidden_activation=hidden_activation,
#             skip_linear=skip_linear,
#             init_strategy=init_strategy,
#             dtype=dtype,
#         )

#         # make aliases for the crosscoder weights with concrete dimensions for this case
#         # according to self.CROSSCODING_DIMS
#         self.W_dec_HMD = self.W_dec_HXD
#         self.W_enc_MDH = self.W_enc_XDH
#         self.b_dec_MD = self.b_dec_XD


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

    @t.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[Any]) -> None:
        assert cc.W_dec_HXD.shape[1] == 2, "expected the model dimension to be 2"

        # do the regular init
        self.first_init.init_weights(cc)

        # BUT: sync the shared decoder weights
        cc.W_dec_HXD[: self.n_shared_latents, 0].copy_(cc.W_dec_HXD[: self.n_shared_latents, 1])

        assert (cc.W_dec_HXD[: self.n_shared_latents, 0] == cc.W_dec_HXD[: self.n_shared_latents, 1]).all()


class BaseDiffingTrainer(Generic[TConfig, TAct]):
    LOG_HISTOGRAMS_EVERY_N_LOGS = 10

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseModelHookpointActivationsDataloader,
        crosscoder: AcausalCrosscoder[TAct],
        n_shared_latents: int,
        wandb_run: Run,
        device: t.device,
        hookpoints: list[str],
        save_dir: Path | str,
    ):
        self.cfg = cfg
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
            self._do_epoch(scaling_factors_M)
            self.epoch += 1
        self.wandb_run.finish()

    def _do_epoch(self, scaling_factors_M: t.Tensor) -> None:
        epoch_dataloader_BMPD = self.activations_dataloader.get_activations_iterator_BMPD()

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
                batch_BMPD = next(epoch_dataloader_BMPD)
                assert batch_BMPD.shape[1] == 2, "we only support 2 models for now"
                assert batch_BMPD.shape[2] == 1, "we only support 1 hookpoint for now"

                batch_BMD = batch_BMPD.squeeze(2).to(self.device)

                train_res = self.crosscoder.forward_train(batch_BMD)
                self.firing_tracker.add_batch(train_res.hidden_BH)

                loss, log_dict = self._loss_and_log_dict(batch_BMD, train_res, log=log)

                loss.div(self.cfg.gradient_accumulation_steps_per_batch).backward()
                if log_dict is not None:
                    log_dicts.append(log_dict)

            if log:
                batch_log_dict_avgs = {
                    **{k: sum(v) / len(v) for k, v in dict_join(log_dicts).items()},
                    **self._step_logs(),
                }
                self.wandb_run.log(batch_log_dict_avgs, step=self.step)

            self._maybe_save_model(scaling_factors_M)

            self._synchronise_shared_weight_grads()
            clip_grad_norm_(self.crosscoder.parameters(), 1.0)
            self._lr_step()
            self.optimizer.step()
            if self.epoch == 0:
                self.unique_tokens_trained += batch_BMD.shape[0]
            self.step += 1

    def _maybe_save_model(self, scaling_factors_M: t.Tensor) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"
            self.crosscoder.with_folded_scaling_factors(scaling_factors_M).save(checkpoint_path)

            if self.cfg.upload_saves_to_wandb and not self.wandb_run.disabled:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

    @abstractmethod
    def _loss_and_log_dict(
        self,
        batch_BMD: t.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[t.Tensor, dict[str, float] | None]: ...

    def _synchronise_shared_weight_grads(self) -> None:
        assert self.crosscoder.W_dec_HXD.grad is not None
        model_0_grad = self.crosscoder.W_dec_HXD.grad[: self.n_shared_latents, 0]
        model_1_grad = self.crosscoder.W_dec_HXD.grad[: self.n_shared_latents, 1]

        summed_grad = model_0_grad + model_1_grad
        model_0_grad.copy_(summed_grad)
        model_1_grad.copy_(summed_grad)

        m0_grads, m1_grads = self.crosscoder.W_dec_HXD.grad[: self.n_shared_latents].unbind(dim=1)
        assert (m0_grads == m1_grads).all()

        m0_weights, m1_weights = self.crosscoder.W_dec_HXD[: self.n_shared_latents].unbind(dim=1)
        assert (m0_weights == m1_weights).all()

    def _lr_step(self) -> None:
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        if self.lr_scheduler is not None:
            self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

    def _step_logs(self) -> dict[str, Any]:
        log_dict = {
            "train/epoch": self.epoch,
            "train/unique_tokens_trained": self.unique_tokens_trained,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if (
            self.cfg.log_every_n_steps is not None
            and self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0
        ):
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.examples_since_fired_A)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})

            W_dec_HiMD = self.crosscoder.W_dec_HXD[self.n_shared_latents :].detach()
            log_dict.update(create_cosine_sim_and_relative_norm_histograms_diffing(W_dec_HMD=W_dec_HiMD))

        return log_dict
    
    def _get_fvu_dict(self, batch_BMD: t.Tensor, recon_acts_BMD: t.Tensor) -> dict[str, float]:
        return get_fvu_dict(batch_BMD, recon_acts_BMD, ("model", [0, 1]))
