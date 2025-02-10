from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import torch as t
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.token_layer_dataloader import BaseTokenLayerActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_trainer import save_config, save_model, validate_num_steps_per_epoch
from model_diffing.scripts.train_jumprelu_sliding_window.run import TokenLayerCrosscoder
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.utils import (
    SaveableModule,
    calculate_explained_variance_X,
    calculate_reconstruction_loss,
    get_explained_var_dict,
    l0_norm,
    l1_norm,
    l2_norm,
    weighted_l1_sparsity_loss,
)

TAct = TypeVar("TAct", bound=SaveableModule)


class BiTokenCCWrapper(nn.Module, Generic[TAct]):
    def __init__(
        self,
        single_token_cc: TokenLayerCrosscoder[TAct],
        double_token_cc: TokenLayerCrosscoder[TAct],
    ):
        super().__init__()

        assert single_token_cc.crosscoding_dims[0] == 1  # token
        assert len(single_token_cc.crosscoding_dims) == 2  # (token, layer)
        self.single_cc = single_token_cc

        assert double_token_cc.crosscoding_dims[0] == 2  # token
        assert len(double_token_cc.crosscoding_dims) == 2  # (token, layer)
        self.double_cc = double_token_cc

    @dataclass
    class TrainResult:
        recon_B1LD_single1: t.Tensor
        recon_B1LD_single2: t.Tensor
        recon_B2LD_double: t.Tensor
        hidden_BH_single1: t.Tensor
        hidden_BH_single2: t.Tensor
        hidden_BH_double: t.Tensor

    def forward_train(self, x_BTLD: t.Tensor) -> TrainResult:
        assert x_BTLD.shape[1] == 2

        output_single1 = self.single_cc.forward_train(x_BTLD[:, 0][:, None])
        output_single2 = self.single_cc.forward_train(x_BTLD[:, 1][:, None])
        output_both = self.double_cc.forward_train(x_BTLD)

        return self.TrainResult(
            recon_B1LD_single1=output_single1.reconstructed_acts_BXD,
            recon_B1LD_single2=output_single2.reconstructed_acts_BXD,
            recon_B2LD_double=output_both.reconstructed_acts_BXD,
            hidden_BH_single1=output_single1.hidden_BH,
            hidden_BH_single2=output_single2.hidden_BH,
            hidden_BH_double=output_both.hidden_BH,
        )

    def forward(self, x_BTLD: t.Tensor) -> t.Tensor:
        return t.Tensor(0)


class L1SlidingWindowCrosscoderTrainer:
    def __init__(
        self,
        cfg: L1TrainConfig,
        activations_dataloader: BaseTokenLayerActivationsDataloader,
        crosscoders: BiTokenCCWrapper[ReLUActivation],
        wandb_run: Run | None,
        device: t.device,
        layers_to_harvest: list[int],
        experiment_name: str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.crosscoders = crosscoders

        self.optimizer = build_optimizer(cfg.optimizer, self.crosscoders.parameters())

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader.num_batches()
        )

        self.total_steps = self.num_steps_per_epoch * (cfg.epochs or 1)
        logger.info(
            f"Total steps: {self.total_steps} (num_steps_per_epoch: {self.num_steps_per_epoch}, epochs: {cfg.epochs})"
        )

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.num_steps_per_epoch)

        self.save_dir = Path(cfg.base_save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        save_config(self.cfg, self.save_dir)

        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            epoch_dataloader_BTLD = self.activations_dataloader.get_shuffled_activations_iterator_BTLD()
            epoch_dataloader_BTLD = islice(epoch_dataloader_BTLD, self.num_steps_per_epoch)

            for batch_BTLD in tqdm(epoch_dataloader_BTLD, desc="Train Steps"):
                batch_BTLD = batch_BTLD.to(self.device)

                self._train_step(batch_BTLD)

                # TODO(oli): get wandb checkpoint saving working

                if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                    scaling_factors_TL = self.activations_dataloader.get_norm_scaling_factors_TL()
                    with self.crosscoders.single_cc.temporarily_fold_activation_scaling(
                        scaling_factors_TL.mean(dim=0, keepdim=True)
                    ):
                        save_model(self.crosscoders.single_cc, self.save_dir / "single_cc", self.epoch, self.step)

                    with self.crosscoders.double_cc.temporarily_fold_activation_scaling(scaling_factors_TL):
                        save_model(self.crosscoders.double_cc, self.save_dir / "double_cc", self.epoch, self.step)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BTLD.shape[0]

                self.step += 1
            self.epoch += 1

    def _train_step(self, batch_BTLD: t.Tensor) -> None:
        self.optimizer.zero_grad()

        # Forward pass
        res = self.crosscoders.forward_train(batch_BTLD)
        reconstructed_acts_BTLD = t.cat([res.recon_B1LD_single1, res.recon_B1LD_single2], dim=1) + res.recon_B2LD_double
        assert reconstructed_acts_BTLD.shape == batch_BTLD.shape, "fuck"

        reconstruction_loss = calculate_reconstruction_loss(batch_BTLD, reconstructed_acts_BTLD)

        sparsity_loss_fn = partial(
            weighted_l1_sparsity_loss,
            layer_reduction=l1_norm,  # encourage layer-level sparsity
            model_reduction=l1_norm,  # sum is a noop on a singleton dim
            token_reduction=l2_norm,  # don't want to encourage token-level sparsity
        )

        sparsity_loss_single1 = sparsity_loss_fn(
            W_dec_HTMLD=self.crosscoders.single_cc.W_dec_HTLD[:, :, None],
            hidden_BH=res.hidden_BH_single1,
        )
        sparsity_loss_single2 = sparsity_loss_fn(
            W_dec_HTMLD=self.crosscoders.single_cc.W_dec_HTLD[:, :, None],
            hidden_BH=res.hidden_BH_single2,
        )
        sparsity_loss_double = sparsity_loss_fn(
            W_dec_HTMLD=self.crosscoders.double_cc.W_dec_HTLD[:, :, None],
            hidden_BH=res.hidden_BH_double,
        )
        sparsity_loss = sparsity_loss_single1 + sparsity_loss_single2 + sparsity_loss_double

        loss = reconstruction_loss + self._lambda_s_scheduler() * sparsity_loss

        # Backward pass
        loss.backward()
        clip_grad_norm_(self.crosscoders.parameters(), 1.0)
        self.optimizer.step()

        # Update learning rate according to scheduler.
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        hidden_B3H = t.cat(
            [res.hidden_BH_single1, res.hidden_BH_double, res.hidden_BH_single2],
            dim=-1,
        )

        if (
            self.cfg.log_every_n_steps is not None
            and self.step % self.cfg.log_every_n_steps == 0
            and self.wandb_run is not None
        ):
            # Instead of building a chart with `get_l0_stats`, we compute and log the values as scalars.
            l0_B = l0_norm(hidden_B3H, dim=-1)
            l0_np = l0_B.detach().cpu().numpy()
            mean_l0 = l0_B.mean().item()
            l0_5, l0_25, l0_75, l0_95 = np.percentile(l0_np, [5, 25, 75, 95])

            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_BTLD, reconstructed_acts_BTLD),
                ("token", [0, 1]),
                ("layer", self.layers_to_harvest),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/lambda_s": self._lambda_s_scheduler(),
                #
                "train/mean_l0_pct": l0_B.mean().item() / hidden_B3H.shape[1],
                # Log grouped l0 statistics as scalars.
                "train/l0/step": self.step,
                "train/l0/5th": l0_5,
                "train/l0/25th": l0_25,
                "train/l0/mean": mean_l0,
                "train/l0/75th": l0_75,
                "train/l0/95th": l0_95,
                #
                "train/loss": loss.item(),
                #
                **explained_variance_dict,
                #
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            self.wandb_run.log(log_dict, step=self.step)

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        if self.step < self.cfg.lambda_s_n_steps:
            return (self.step / self.cfg.lambda_s_n_steps) * self.cfg.lambda_s_max

        return self.cfg.lambda_s_max
