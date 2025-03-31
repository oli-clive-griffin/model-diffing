from abc import abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import torch
from wandb.sdk.wandb_run import Run

from crosscoding.data.activations_dataloader import ModelHookpointActivationsBatch, ModelHookpointActivationsDataloader
from crosscoding.models.activations.activation_function import ActivationFunction
from crosscoding.models.sparse_coders import ModelHookpointAcausalCrosscoder
from crosscoding.trainers.base_trainer import BaseTrainer
from crosscoding.trainers.config_common import BaseTrainConfig
from crosscoding.trainers.utils import wandb_histogram
from crosscoding.trainers.wandb_utils.main import create_checkpoint_artifact
from crosscoding.utils import get_fvu_dict

TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=ActivationFunction)


class BaseModelHookpointAcausalTrainer(
    BaseTrainer[TConfig, ModelHookpointAcausalCrosscoder[TAct], ModelHookpointActivationsBatch]
):
    def __init__(
        self,
        cfg: TConfig,
        n_models: int,
        hookpoints: list[str],
        activations_dataloader: ModelHookpointActivationsDataloader,
        crosscoder: ModelHookpointAcausalCrosscoder[TAct],
        wandb_run: Run,
        device: torch.device,
        save_dir: Path | str,
        # crosscoding_dims: CrosscodingDimsDict,
    ):
        super().__init__(cfg, activations_dataloader, crosscoder, wandb_run, device, save_dir)

        self.n_models = n_models
        self.hookpoints = hookpoints

        # self.crosscoding_dims = crosscoding_dims

        # assert (
        #     self.crosscoding_dims
        #     == self.crosscoder.crosscoding_dims
        #     == self.activations_dataloader.get_crosscoding_dims()
        # ), "The crosscoder must have the same crosscoding dims as the activations dataloader"

    def run_batch(
        self, batch: ModelHookpointActivationsBatch, log: bool
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        train_res = self.crosscoder.forward_train(batch.activations_BMPD.to(self.device))
        self.firing_tracker.add_batch(train_res.latents_BL)
        return self._calculate_loss_and_log(batch.activations_BMPD, train_res, log=log)

    def _maybe_save_model(self) -> None:
        if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
            checkpoint_path = self.save_dir / f"epoch_{self.epoch}_step_{self.step}"

            scaling_factors_MP = self.activations_dataloader.get_scaling_factors().to(self.device)
            self.crosscoder.with_folded_scaling_factors(scaling_factors_MP, scaling_factors_MP).save(checkpoint_path)

            if self.cfg.upload_saves_to_wandb:
                artifact = create_checkpoint_artifact(checkpoint_path, self.wandb_run.id, self.step, self.epoch)
                self.wandb_run.log_artifact(artifact)

    @abstractmethod
    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]: ...

    def _step_logs(self) -> dict[str, Any]:
        log_dict = super()._step_logs()
        if self.step % (self.cfg.log_every_n_steps * self.LOG_HISTOGRAMS_EVERY_N_LOGS) == 0:
            tokens_since_fired_hist = wandb_histogram(self.firing_tracker.tokens_since_fired_L)
            log_dict.update({"media/tokens_since_fired": tokens_since_fired_hist})

            # if (model_dim := self.crosscoding_dims.get("model")) is not None and len(model_dim) == 2:
            #     other_cc_dims = [dim for dim in self.crosscoding_dims.values() if dim.name != "model"]
            #     sets_of_label_idx_pairs = [list(enumerate(dim.index_labels)) for dim in other_cc_dims]
            #     for accessors in itertools.product(*sets_of_label_idx_pairs):
            #         indices = [idx for idx, _ in accessors]
            #         labels = [label for _, label in accessors]

            #         W_dec_LXD = self.crosscoder.W_dec_LXD.detach().cpu()
            #         W_dec_LXD = W_dec_LXD.select(dim=model_dim.index_labels, index=indices)
            #         relative_decoder_norms_plot, shared_features_cosine_sims_plot = (
            #             create_cosine_sim_and_relative_norm_histograms(self.crosscoder.W_dec_LXD.detach().cpu())
            #         )
            #         log_dict.update(
            #             {
            #                 "media/relative_decoder_norms": relative_decoder_norms_plot,
            #                 "media/shared_features_cosine_sims": shared_features_cosine_sims_plot,
            #             }
            #         )

        return log_dict

    def _get_fvu_dict(self, batch_BMPD: torch.Tensor, recon_acts_BMPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            batch_BMPD,
            recon_acts_BMPD,
            # *((dim.name, dim.index_labels) for dim in self.crosscoding_dims.values()),
            ("model", ["0", "1"]),
            ("hookpoint", self.hookpoints),
        )
