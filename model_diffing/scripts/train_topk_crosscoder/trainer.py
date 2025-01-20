from dataclasses import dataclass

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase
from wandb.sdk.wandb_run import Run

from model_diffing.dataloader.data import ShuffledTokensActivationsLoader
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.utils import estimate_norm_scaling_factor
from model_diffing.utils import reconstruction_loss, save_model_and_config

from .config import TrainConfig


@dataclass
class TopKLossInfo:
    reconstruction_loss: float
    l0: float


class TopKTrainer:
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
        self.d_model = self.llms[0].cfg.d_model

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

    def train(self):
        dataloader_iterator_BMLD = self.dataloader.get_shuffled_activations_iterator_BMLD()

        norm_scaling_factor = estimate_norm_scaling_factor(
            dataloader_iterator_BMLD,
            self.d_model,
            self.cfg.n_batches_for_norm_estimate,
        )

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )

        while self.step < self.cfg.num_steps:
            batch_BMLD = next(dataloader_iterator_BMLD)
            batch_BMLD = batch_BMLD.to(self.device)
            batch_BMLD = batch_BMLD * norm_scaling_factor
            self.train_step(batch_BMLD)
            self.step += 1

    def get_loss(self, activations_BMLD: torch.Tensor) -> tuple[torch.Tensor, TopKLossInfo]:
        train_res = self.crosscoder.forward_train(activations_BMLD)

        reconstruction_loss_ = reconstruction_loss(activations_BMLD, train_res.reconstructed_acts_BMLD)

        loss_info = TopKLossInfo(
            reconstruction_loss=reconstruction_loss_.item(),
            l0=(train_res.hidden_BH > 0).float().sum().item(),
        )

        return reconstruction_loss_, loss_info

    def train_step(self, batch_BMLD: torch.Tensor):
        self.optimizer.zero_grad()

        loss, loss_info = self.get_loss(batch_BMLD)

        loss.backward()

        if (self.step + 1) % self.cfg.log_every_n_steps == 0:
            log_dict = {
                "train/step": self.step,
                "train/l0": loss_info.l0,
                "train/reconstruction_loss": loss_info.reconstruction_loss,
                "train/loss": loss.item(),
            }
            print(log_dict)
            if self.wandb_run:
                self.wandb_run.log(log_dict)

        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] = self._lr_scheduler()

        if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
            save_model_and_config(
                config=self.cfg,
                save_dir=self.cfg.save_dir,
                model=self.crosscoder,
                epoch=self.step,
            )

    def _lr_scheduler(self) -> float:
        pct_until_finished = 1 - (self.step / self.cfg.num_steps)
        if pct_until_finished < self.cfg.learning_rate.last_pct_of_steps:
            # 1 at the last step of constant learning rate period
            # 0 at the end of training
            scale = pct_until_finished / self.cfg.learning_rate.last_pct_of_steps
            return self.cfg.learning_rate.initial_learning_rate * scale
        else:
            return self.cfg.learning_rate.initial_learning_rate
