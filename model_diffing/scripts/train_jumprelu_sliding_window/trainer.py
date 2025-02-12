from itertools import islice
from pathlib import Path
from typing import Any

import torch as t
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.data.token_hookpoint_dataloader import BaseTokenhookpointActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.activations.jumprelu import JumpReLUActivation
from model_diffing.scripts.base_trainer import save_config, save_model, validate_num_steps_per_epoch
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateTrainConfig
from model_diffing.scripts.train_l1_sliding_window.trainer import BiTokenCCWrapper
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer, wandb_histogram
from model_diffing.utils import (
    calculate_explained_variance_X,
    calculate_reconstruction_loss,
    get_decoder_norms_H,
    get_explained_var_dict,
    l0_norm,
)


class JumpreluSlidingWindowCrosscoderTrainer:
    def __init__(
        self,
        cfg: JanUpdateTrainConfig,
        activations_dataloader: BaseTokenhookpointActivationsDataloader,
        crosscoders: BiTokenCCWrapper[JumpReLUActivation],
        wandb_run: Run | None,
        device: t.device,
        hookpoints: list[str],
        experiment_name: str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.hookpoints = hookpoints

        self.crosscoders = crosscoders

        self.optimizer = build_optimizer(cfg.optimizer, self.crosscoders.parameters())

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader.num_batches()
        )

        self.total_steps = self.num_steps_per_epoch * (cfg.epochs or 1)
        logger.info(
            f"Total steps: {self.total_steps} (num_steps_per_epoch: {self.num_steps_per_epoch}, epochs: {cfg.epochs})"
        )

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.total_steps)

        self.save_dir = Path(cfg.base_save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        save_config(self.cfg, self.save_dir)

        epoch_iter = tqdm(range(self.cfg.epochs), desc="Epochs") if self.cfg.epochs is not None else range(1)
        for _ in epoch_iter:
            epoch_dataloader_BTPD = self.activations_dataloader.get_shuffled_activations_iterator_BTPD()
            epoch_dataloader_BTPD = islice(epoch_dataloader_BTPD, self.num_steps_per_epoch)

            for batch_BTPD in tqdm(epoch_dataloader_BTPD, desc="Train Steps"):
                batch_BTPD = batch_BTPD.to(self.device)

                self._train_step(batch_BTPD)

                if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                    scaling_factors_TP = self.activations_dataloader.get_norm_scaling_factors_TP()
                    with self.crosscoders.single_cc.temporarily_fold_activation_scaling(
                        scaling_factors_TP.mean(dim=0, keepdim=True)
                    ):
                        save_model(self.crosscoders.single_cc, self.save_dir / "single_cc", self.epoch, self.step)

                    with self.crosscoders.double_cc.temporarily_fold_activation_scaling(scaling_factors_TP):
                        save_model(self.crosscoders.double_cc, self.save_dir / "double_cc", self.epoch, self.step)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BTPD.shape[0]

                self.step += 1
            self.epoch += 1

    def _train_step(self, batch_BTPD: t.Tensor) -> None:
        self.optimizer.zero_grad()

        # fwd
        res = self.crosscoders.forward_train(batch_BTPD)

        reconstructed_acts_BTPD = t.cat([res.recon_B1PD_single1, res.recon_B1PD_single2], dim=1) + res.recon_B2PD_double
        assert reconstructed_acts_BTPD.shape == batch_BTPD.shape, "fuck"

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BTPD, reconstructed_acts_BTPD)

        hidden_B3H = t.cat([res.hidden_BH_single1, res.hidden_BH_double, res.hidden_BH_single2], dim=-1)

        decoder_norms_single_H = get_decoder_norms_H(self.crosscoders.single_cc.W_dec_HXD)
        decoder_norms_both_H = get_decoder_norms_H(self.crosscoders.double_cc.W_dec_HXD)

        decoder_norms_3H = t.cat([decoder_norms_single_H, decoder_norms_both_H, decoder_norms_single_H], dim=-1)

        tanh_sparsity_loss = self._tanh_sparsity_loss(hidden_B3H, decoder_norms_3H)
        pre_act_loss = self._pre_act_loss(hidden_B3H, decoder_norms_3H)

        lambda_s = self._lambda_s_scheduler()
        scaled_tanh_sparsity_loss = lambda_s * tanh_sparsity_loss
        scaled_pre_act_loss = self.cfg.lambda_p * pre_act_loss

        loss = (
            reconstruction_loss  #
            + scaled_tanh_sparsity_loss
            + scaled_pre_act_loss
        )

        # backward
        loss.backward()
        clip_grad_norm_(self.crosscoders.parameters(), 1.0)
        self.optimizer.step()

        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and (self.step + 1) % self.cfg.log_every_n_steps == 0
        ):
            mean_l0 = l0_norm(hidden_B3H, dim=-1).mean()

            thresholds_single_hist = wandb_histogram(self.crosscoders.single_cc.hidden_activation.log_threshold_H.exp())

            thresholds_both_hist = wandb_histogram(self.crosscoders.double_cc.hidden_activation.log_threshold_H.exp())

            with t.no_grad():
                from einops import einsum

                cc1_t1_pre_biases_BH = einsum(
                    batch_BTPD[:, 0][:, None], self.crosscoders.single_cc.W_enc_XDH, "b t p d, t p d h -> b h"
                )
                cc1_t1_pre_biases_hist = wandb_histogram(cc1_t1_pre_biases_BH.flatten())

                cc1_t2_pre_biases_BH = einsum(
                    batch_BTPD[:, 1][:, None], self.crosscoders.single_cc.W_enc_XDH, "b t p d, t p d h -> b h"
                )
                cc1_t2_pre_biases_hist = wandb_histogram(cc1_t2_pre_biases_BH.flatten())

                cc2_pre_biases_BH = einsum(batch_BTPD, self.crosscoders.double_cc.W_enc_XDH, "b t p d, t p d h -> b h")
                cc2_pre_biases_hist = wandb_histogram(cc2_pre_biases_BH.flatten())

                activations_hist = wandb_histogram(hidden_B3H)

            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_BTPD, reconstructed_acts_BTPD),
                ("token", [0, 1]),
                ("hookpoint", self.hookpoints),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                #
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/tanh_sparsity_loss_scaled": scaled_tanh_sparsity_loss.item(),
                "train/lambda_s": lambda_s,
                #
                "train/mean_l0": mean_l0.item(),
                "train/mean_l0_pct": mean_l0.item() / hidden_B3H.shape[1],
                #
                "train/pre_act_loss": pre_act_loss.item(),
                "train/pre_act_loss_scaled": scaled_pre_act_loss.item(),
                "train/lambda_p": self.cfg.lambda_p,
                #
                "train/loss": loss.item(),
                #
                **explained_variance_dict,
                #
                "media/jumprelu_threshold_distribution_single": thresholds_single_hist,
                "media/jumprelu_threshold_distribution_both": thresholds_both_hist,
                # "media/pre_bias_distribution": pre_biases_hist,
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                #
                "media/cc1_t1_pre_biases": cc1_t1_pre_biases_hist,
                "media/cc1_t2_pre_biases": cc1_t2_pre_biases_hist,
                "media/cc2_pre_biases": cc2_pre_biases_hist,
                "media/activations": activations_hist,
            }

            self.wandb_run.log(log_dict, step=self.step)

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        loss_BH = t.tanh(self.cfg.c * hidden_BH * decoder_norms_H)
        return loss_BH.sum(-1).mean()

    def _pre_act_loss(self, hidden_B3H: t.Tensor, decoder_norms_3H: t.Tensor) -> t.Tensor:
        t_3H = t.cat(
            [
                self.crosscoders.single_cc.hidden_activation.log_threshold_H,
                self.crosscoders.double_cc.hidden_activation.log_threshold_H,
                self.crosscoders.single_cc.hidden_activation.log_threshold_H,
            ],
            dim=-1,
        )
        loss_3H = t.relu(t_3H.exp() - hidden_B3H) * decoder_norms_3H
        return loss_3H.sum(-1).mean()
