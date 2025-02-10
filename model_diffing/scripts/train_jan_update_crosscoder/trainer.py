from typing import Any

import torch as t
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.jumprelu import JumpReLUActivation
from model_diffing.scripts.base_trainer import BaseModelLayerTrainer
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateTrainConfig
from model_diffing.scripts.utils import create_cosine_sim_and_relative_norm_histograms, get_l0_stats, wandb_histogram
from model_diffing.utils import (
    calculate_explained_variance_X,
    calculate_reconstruction_loss,
    get_decoder_norms_H,
    get_explained_var_dict,
    l0_norm,
)


class JanUpdateCrosscoderTrainer(BaseModelLayerTrainer[JanUpdateTrainConfig, JumpReLUActivation]):
    def _train_step(self, batch_BMLD: t.Tensor) -> None:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMLD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BXD)

        decoder_norms_H = get_decoder_norms_H(self.crosscoder.W_dec_HXD)
        tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.hidden_BH, decoder_norms_H)
        pre_act_loss = self._pre_act_loss(train_res.hidden_BH, decoder_norms_H)

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
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and self.step % self.cfg.log_every_n_steps == 0
        ):
            l0_B = l0_norm(train_res.hidden_BH, dim=-1)

            l0_stats = get_l0_stats(l0_B, self.step)

            thresholds_hist = wandb_histogram(self.crosscoder.hidden_activation.log_threshold_H.exp())

            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_BMLD, train_res.reconstructed_acts_BXD),
                ("model", list(range(self.n_models))),
                ("layer", self.layers_to_harvest),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                #
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/tanh_sparsity_loss_scaled": scaled_tanh_sparsity_loss.item(),
                "train/lambda_s": lambda_s,
                #
                "train/mean_l0_pct": l0_B.mean().item() / self.crosscoder.hidden_dim,
                #
                "train/pre_act_loss": pre_act_loss.item(),
                "train/pre_act_loss_scaled": scaled_pre_act_loss.item(),
                "train/lambda_p": self.cfg.lambda_p,
                #
                "train/loss": loss.item(),
                #
                "media/jumprelu_threshold_distribution": thresholds_hist,
                #
                **explained_variance_dict,
                **l0_stats,
            }

            if self.n_models == 2:
                W_dec_HXD = self.crosscoder.W_dec_HXD.detach().cpu()
                assert W_dec_HXD.shape[1:-1] == (self.n_models, self.n_layers)
                log_dict.update(
                    create_cosine_sim_and_relative_norm_histograms(
                        W_dec_HMLD=W_dec_HXD,
                        layers=self.layers_to_harvest,
                    )
                )

            self.wandb_run.log(log_dict, step=self.step)

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        loss_BH = t.tanh(self.cfg.c * hidden_BH * decoder_norms_H)
        return loss_BH.sum(-1).mean()

    def _pre_act_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        t_H = self.crosscoder.hidden_activation.log_threshold_H
        loss_BH = t.relu(t_H.exp() - hidden_BH) * decoder_norms_H
        return loss_BH.sum(-1).mean()
