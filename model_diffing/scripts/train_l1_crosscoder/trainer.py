from dataclasses import dataclass

import torch
import wandb
from einops import einsum
from torch.nn.utils import clip_grad_norm_
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase
from wandb.sdk.wandb_run import Run

from model_diffing.dataloader.data import ShuffledTokensActivationsLoader
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.utils import estimate_norm_scaling_factor_ML
from model_diffing.utils import reconstruction_loss, save_model_and_config, sparsity_loss_l1_of_norms

from .config import TrainConfig


@dataclass
class L1LossInfo:
    lambda_: float
    reconstruction_loss: float
    sparsity_loss: float
    l0: float


class L1SaeTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        llms: list[HookedTransformer],
        optimizer: torch.optim.Optimizer,
        dataloader: ShuffledTokensActivationsLoader,
        crosscoder: AcausalCrosscoder,
        wandb_run: Run | None,
        device: torch.device,
        # hacky - remove:
        expected_batch_shape: tuple[int, int, int, int],
    ):
        self.cfg = cfg
        self.llms = llms

        # assert all(llm.tokenizer == llms[0].tokenizer for llm in llms), (
        #     "All models must have the same tokenizer"
        # )
        tokenizer = self.llms[0].tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        self.tokenizer = tokenizer
        self.crosscoder = crosscoder
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.wandb_run = wandb_run
        self.expected_batch_shape = expected_batch_shape
        self.device = device

        self.step = 0
        self.dataloader_iterator_BMLD = self.dataloader.get_shuffled_activations_iterator_BMLD()

    @property
    def num_models(self) -> int:
        return len(self.llms)

    @property
    def num_layers(self) -> int:
        return self.llms[0].cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.llms[0].cfg.d_model

    def train(self):
        norm_scaling_factors_ML = self._estimate_norm_scaling_factor_ML()

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )

        while self.step < self.cfg.num_steps:
            batch_BMLD = self._next_batch_BMLD(norm_scaling_factors_ML)

            log_dict = self._train_step(batch_BMLD)

            if (self.step + 1) % self.cfg.log_every_n_steps == 0:
                logger.info(log_dict)
                if self.wandb_run:
                    self.wandb_run.log(log_dict)

            if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
                save_model_and_config(
                    config=self.cfg,
                    save_dir=self.cfg.save_dir,
                    model=self.crosscoder,
                    epoch=self.step,
                )

            self.step += 1

    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        loss, loss_info = self._get_loss(batch_BMLD)

        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()

        self.optimizer.param_groups[0]["lr"] = self._lr_scheduler()

        log_dict = {
            "train/step": self.step,
            "train/lambda": loss_info.lambda_,
            "train/l0": loss_info.l0,
            "train/reconstruction_loss": loss_info.reconstruction_loss,
            "train/sparsity_loss": loss_info.sparsity_loss,
            "train/loss": loss.item(),
        }

        return log_dict

    def _get_loss(self, activations_BMLD: torch.Tensor) -> tuple[torch.Tensor, "L1LossInfo"]:
        train_res = self.crosscoder.forward_train(activations_BMLD)

        reconstruction_loss_ = reconstruction_loss(activations_BMLD, train_res.reconstructed_acts_BMLD)
        sparsity_loss_ = sparsity_loss_l1_of_norms(self.crosscoder.W_dec_HMLD, train_res.hidden_BH)
        lambda_ = self._l1_coef_scheduler()

        loss = reconstruction_loss_ + lambda_ * sparsity_loss_

        loss_info = L1LossInfo(
            lambda_=lambda_,
            reconstruction_loss=reconstruction_loss_.item(),
            sparsity_loss=sparsity_loss_.item(),
            l0=(train_res.hidden_BH > 0).float().sum().item(),
        )

        return loss, loss_info

    def _estimate_norm_scaling_factor_ML(self) -> torch.Tensor:
        return estimate_norm_scaling_factor_ML(
            self.dataloader_iterator_BMLD,
            self.num_models,
            self.num_layers,
            self.d_model,
            self.cfg.n_batches_for_norm_estimate,
        )

    def _next_batch_BMLD(self, norm_scaling_factors_ML: torch.Tensor) -> torch.Tensor:
        batch_BMLD = next(self.dataloader_iterator_BMLD)
        batch_BMLD = batch_BMLD.to(self.device)
        batch_BMLD = einsum(
            batch_BMLD, norm_scaling_factors_ML, "batch model layer d_model, model layer -> batch model layer d_model"
        )
        return batch_BMLD

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.lambda_n_steps:
            return self.cfg.lambda_max * self.step / self.cfg.lambda_n_steps
        else:
            return self.cfg.lambda_max

    def _lr_scheduler(self) -> float:
        pct_until_finished = 1 - (self.step / self.cfg.num_steps)
        if pct_until_finished < self.cfg.learning_rate.last_pct_of_steps:
            # 1 at the last step of constant learning rate period
            # 0 at the end of training
            scale = pct_until_finished / self.cfg.learning_rate.last_pct_of_steps
            return self.cfg.learning_rate.initial_learning_rate * scale
        else:
            return self.cfg.learning_rate.initial_learning_rate
