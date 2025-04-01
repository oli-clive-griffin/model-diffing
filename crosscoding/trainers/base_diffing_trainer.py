from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from crosscoding.data.activations_dataloader import ModelHookpointActivationsDataloader
from crosscoding.models.activations.activation_function import ActivationFunction
from crosscoding.models.initialization.init_strategy import InitStrategy
from crosscoding.models.sparse_coders import ModelHookpointAcausalCrosscoder
from crosscoding.trainers.base_acausal_trainer import BaseModelHookpointAcausalTrainer
from crosscoding.trainers.config_common import BaseTrainConfig
from crosscoding.trainers.utils import create_cosine_sim_and_relative_norm_histograms_diffing

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class IdenticalLatentsInit(InitStrategy[ModelHookpointAcausalCrosscoder[Any]]):
    """
    Init strategy that first applies a regular init, and then sets the decoder weight such that each model
    has the same shared decoder weights for the first n_shared_latents.
    """

    def __init__(
        self,
        first_init: InitStrategy[ModelHookpointAcausalCrosscoder[Any]],
        n_shared_latents: int,
    ):
        self.first_init = first_init
        self.n_shared_latents = n_shared_latents

    @torch.no_grad()
    def init_weights(self, cc: ModelHookpointAcausalCrosscoder[Any]) -> None:
        assert cc.W_dec_LMPD.shape[1] == 2, "expected the model dimension to be 2"

        # do the regular init
        self.first_init.init_weights(cc)

        # BUT: sync the shared decoder weights
        cc.W_dec_LMPD[: self.n_shared_latents, 0].copy_(cc.W_dec_LMPD[: self.n_shared_latents, 1])

        assert (cc.W_dec_LMPD[: self.n_shared_latents, 0] == cc.W_dec_LMPD[: self.n_shared_latents, 1]).all()


class BaseFebUpdateDiffingTrainer(Generic[TConfig, TAct], BaseModelHookpointAcausalTrainer[TConfig, TAct]):
    activations_dataloader: ModelHookpointActivationsDataloader

    def __init__(
        self,
        cfg: TConfig,
        hookpoints: list[str],
        activations_dataloader: ModelHookpointActivationsDataloader,
        crosscoder: ModelHookpointAcausalCrosscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
        n_shared_latents: int,
    ):
        n_models = 2
        super().__init__(cfg, n_models, hookpoints, activations_dataloader, crosscoder, wandb_run, device, save_dir)
        self.n_shared_latents = n_shared_latents
        assert self.crosscoder.n_models == 2, "expected the model crosscoding dim to have length 2"
        assert self.activations_dataloader.n_models == 2, "expected the activations dataloader to have length 2"

    def _after_forward_passes(self):
        self._synchronise_shared_weight_grads()

    def _synchronise_shared_weight_grads(self) -> None:
        assert self.crosscoder.W_dec_LMPD.grad is not None
        W_dec_grad_LMPD = self.crosscoder.W_dec_LMPD.grad[: self.n_shared_latents]

        model_0_grad_LPD = W_dec_grad_LMPD[:, 0]
        model_1_grad_LPD = W_dec_grad_LMPD[:, 1]

        summed_grad = model_0_grad_LPD + model_1_grad_LPD
        model_0_grad_LPD.copy_(summed_grad)
        model_1_grad_LPD.copy_(summed_grad)

        m0_grads, m1_grads = self.crosscoder.W_dec_LMPD.grad[: self.n_shared_latents].unbind(dim=1)
        assert (m0_grads == m1_grads).all()

        m0_weights, m1_weights = self.crosscoder.W_dec_LMPD[: self.n_shared_latents].unbind(dim=1)
        assert (m0_weights == m1_weights).all()

    def _step_logs(self) -> dict[str, Any]:
        log_dict = super()._step_logs()

        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            W_dec_LiMD = self.crosscoder.W_dec_LMPD[self.n_shared_latents :].detach()
            log_dict.update(create_cosine_sim_and_relative_norm_histograms_diffing(W_dec_LMD=W_dec_LiMD))

        return log_dict

