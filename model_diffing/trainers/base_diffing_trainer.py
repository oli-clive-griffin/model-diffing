from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.models.crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.trainers.base_trainer import BaseTrainer
from model_diffing.trainers.config_common import BaseTrainConfig
from model_diffing.trainers.utils import (
    create_cosine_sim_and_relative_norm_histograms_diffing,
    wandb_histogram,
)
from model_diffing.trainers.wandb_utils.main import create_checkpoint_artifact
from model_diffing.utils import get_fvu_dict

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
        assert cc.W_dec_LXD.shape[1] == 2, "expected the model dimension to be 2"

        # do the regular init
        self.first_init.init_weights(cc)

        # BUT: sync the shared decoder weights
        cc.W_dec_LXD[: self.n_shared_latents, 0].copy_(cc.W_dec_LXD[: self.n_shared_latents, 1])

        assert (cc.W_dec_LXD[: self.n_shared_latents, 0] == cc.W_dec_LXD[: self.n_shared_latents, 1]).all()


class BaseDiffingTrainer(BaseTrainer[TConfig, AcausalCrosscoder[TAct]], Generic[TConfig, TAct]):
    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseModelHookpointActivationsDataloader,
        crosscoder: Any,
        wandb_run: Run,
        device: torch.device,
        hookpoints: list[str],
        save_dir: Path | str,
        n_shared_latents: int,
    ):
        super().__init__(cfg, activations_dataloader, crosscoder, wandb_run, device, hookpoints, save_dir)
        self.n_shared_latents = n_shared_latents

    def run_batch(self, batch_BMPD: torch.Tensor, log: bool) -> tuple[torch.Tensor, dict[str, float] | None]:
        assert batch_BMPD.shape[1] == 2, "we only support 2 models for now"
        assert batch_BMPD.shape[2] == 1, "we only support 1 hookpoint for now"

        batch_BMD = batch_BMPD.squeeze(2).to(self.device)

        train_res = self.crosscoder.forward_train(batch_BMD)
        self.firing_tracker.add_batch(train_res.latents_BL)

        return self._loss_and_log_dict(batch_BMD, train_res, log=log)

    @abstractmethod
    def _loss_and_log_dict(
        self,
        batch_BMD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def _maybe_save_model(self, scaling_factors_MP: torch.Tensor) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"
            self.crosscoder.with_folded_scaling_factors(scaling_factors_MP, scaling_factors_MP).save(checkpoint_path)

            if self.cfg.upload_saves_to_wandb and not self.wandb_run.disabled:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

    def _after_forward_passes(self):
        self._synchronise_shared_weight_grads()

    def _synchronise_shared_weight_grads(self) -> None:
        assert self.crosscoder.W_dec_LXD.grad is not None
        model_0_grad = self.crosscoder.W_dec_LXD.grad[: self.n_shared_latents, 0]
        model_1_grad = self.crosscoder.W_dec_LXD.grad[: self.n_shared_latents, 1]

        summed_grad = model_0_grad + model_1_grad
        model_0_grad.copy_(summed_grad)
        model_1_grad.copy_(summed_grad)

        m0_grads, m1_grads = self.crosscoder.W_dec_LXD.grad[: self.n_shared_latents].unbind(dim=1)
        assert (m0_grads == m1_grads).all()

        m0_weights, m1_weights = self.crosscoder.W_dec_LXD[: self.n_shared_latents].unbind(dim=1)
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
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.tokens_since_fired_L)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})

            W_dec_LiMD = self.crosscoder.W_dec_LXD[self.n_shared_latents :].detach()
            log_dict.update(create_cosine_sim_and_relative_norm_histograms_diffing(W_dec_LMD=W_dec_LiMD))

            # log bias histograms?

        return log_dict

    def _get_fvu_dict(self, batch_BMD: torch.Tensor, recon_acts_BMD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(batch_BMD, recon_acts_BMD, ("model", [0, 1]))
