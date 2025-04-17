from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.jumprelu import AnthropicSTEJumpReLUActivation
from crosscode.trainers.firing_tracker import FiringTracker
from crosscode.trainers.jan_update_crosscoder.trainer import pre_act_loss, tanh_sparsity_loss
from crosscode.trainers.trainer import ModelWrapper
from crosscode.trainers.utils import create_cosine_sim_and_relative_norm_histograms, get_l0_stats, wandb_histogram
from crosscode.utils import calculate_reconstruction_loss_summed_norm_MSEs, get_fvu_dict, get_summed_decoder_norms_L


class JumpReLUModelDiffingFebUpdateWrapper(ModelWrapper):
    """
    Implementation of https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html but with jumprelu
    loss as described in https://transformer-circuits.pub/2025/january-update/index.html.
    """

    def __init__(
        self,
        model: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation],
        n_shared_latents: int,
        lambda_p: float,
        num_steps: int,
        final_lambda_s: float,
        final_lambda_f: float,
        c: float,
        model_names: list[str],
        hookpoints: list[str],
        save_dir: Path,
        scaling_factors_MP: torch.Tensor,
    ):
        assert model.n_models == 2, "we should have two models"
        assert len(model_names) == 2, "we should have two model names"

        self.crosscoder = model
        self.n_shared_latents = n_shared_latents
        self.num_steps = num_steps
        self.lambda_p = lambda_p
        self.final_lambda_s = final_lambda_s
        self.final_lambda_f = final_lambda_f
        self.model_names = model_names
        self.hookpoints = hookpoints
        self.save_dir = save_dir
        self.c = c

        assert scaling_factors_MP.shape[0] == 2, "we should have two models in the scaling factors"
        self.scaling_factors_MP = scaling_factors_MP

        self.firing_tracker = FiringTracker(activation_size=model.n_latents, device=self.crosscoder.device)

    def _synchronise_shared_latents_gradients(self) -> None:
        """
        Synchronise the gradients of the shared latents across the two models.
        """
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

    def run_batch(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        n_models = batch.activations_BMPD.shape[1]
        assert n_models == 2, "we should have two models"

        train_res = self.crosscoder.forward_train(batch.activations_BMPD)
        self.firing_tracker.add_batch(train_res.latents_BL)

        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(
            batch.activations_BMPD, train_res.recon_acts_BMPD
        )

        decoder_norms_L = get_summed_decoder_norms_L(self.crosscoder.W_dec_LMPD)
        decoder_norms_shared_Ls = decoder_norms_L[: self.n_shared_latents]
        decoder_norms_indep_Li = decoder_norms_L[self.n_shared_latents :]

        hidden_shared_BLs = train_res.latents_BL[:, : self.n_shared_latents]
        hidden_indep_BLi = train_res.latents_BL[:, self.n_shared_latents :]

        # shared features sparsity loss
        tanh_sparsity_loss_shared = self._tanh_sparsity_loss(hidden_shared_BLs, decoder_norms_shared_Ls)
        lambda_s = self._lambda_s_scheduler(step)

        # independent features sparsity loss
        tanh_sparsity_loss_indep = self._tanh_sparsity_loss(hidden_indep_BLi, decoder_norms_indep_Li)
        lambda_f = self._lambda_f_scheduler(step)

        # pre-activation loss on all features
        pre_act_loss = self._pre_act_loss(train_res.latents_BL, decoder_norms_L)

        # total loss
        loss = (
            reconstruction_loss  #
            + lambda_s * tanh_sparsity_loss_shared
            + lambda_f * tanh_sparsity_loss_indep
            + self.lambda_p * pre_act_loss
        )

        if log:
            hidden_shared_BLs = train_res.latents_BL[:, : self.n_shared_latents]
            hidden_indep_BLi = train_res.latents_BL[:, self.n_shared_latents :]

            log_dict: dict[str, float] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss_shared": tanh_sparsity_loss_shared.item(),
                "train/tanh_sparsity_loss_indep": tanh_sparsity_loss_indep.item(),
                "train/pre_act_loss": pre_act_loss.item(),
                "train/loss": loss.item(),
                "train/lambda_s": lambda_s,
                "train/lambda_f": lambda_f,
                **self._get_fvu_dict(batch.activations_BMPD, train_res.recon_acts_BMPD),
                **get_l0_stats(hidden_shared_BLs, name="shared_l0"),
                **get_l0_stats(hidden_indep_BLi, name="indep_l0"),
                **get_l0_stats(train_res.latents_BL, name="both_l0"),
            }
            return loss, log_dict

        return loss, None

    def save(self, step: int) -> Path:
        checkpoint_path = self.save_dir / f"step_{step}"
        self.crosscoder.with_folded_scaling_factors(self.scaling_factors_MP).save(checkpoint_path)
        return checkpoint_path

    def _get_fvu_dict(self, y_BMPD: torch.Tensor, recon_y_BMPD: torch.Tensor) -> dict[str, float]:
        return get_fvu_dict(
            y_BMPD,
            recon_y_BMPD,
            ("hookpoint", self.hookpoints),
            ("model", self.model_names),
        )

    LOG_HISTOGRAMS_EVERY_N_LOGS = 10

    def expensive_logs(self) -> dict[str, Any]:
        log_dict: dict[str, Any] = {
            "media/jr_threshold": wandb_histogram(self.crosscoder.activation_fn.log_threshold_L.exp()),
        }

        if self.crosscoder.n_models == 2:
            W_dec_LMPD = self.crosscoder.W_dec_LMPD[self.n_shared_latents :].detach()  # .cpu()
            for p, hookpoint in enumerate(self.hookpoints):
                relative_decoder_norms_plot, shared_features_cosine_sims_plot = (
                    create_cosine_sim_and_relative_norm_histograms(W_dec_LMD=W_dec_LMPD[:, :, p])
                )
                if relative_decoder_norms_plot is not None:
                    log_dict[f"media/relative_decoder_norms_{hookpoint}"] = relative_decoder_norms_plot
                if shared_features_cosine_sims_plot is not None:
                    log_dict[f"media/shared_features_cosine_sims_{hookpoint}"] = shared_features_cosine_sims_plot

        return log_dict

    def _lambda_s_scheduler(self, step: int) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (step / self.num_steps) * self.final_lambda_s

    def _lambda_f_scheduler(self, step: int) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (step / self.num_steps) * self.final_lambda_f

    def _tanh_sparsity_loss(self, hidden_BL: torch.Tensor, decoder_norms_L: torch.Tensor) -> torch.Tensor:
        return tanh_sparsity_loss(self.c, hidden_BL, decoder_norms_L)

    def _pre_act_loss(self, hidden_BL: torch.Tensor, decoder_norms_L: torch.Tensor) -> torch.Tensor:
        return pre_act_loss(self.crosscoder.activation_fn.log_threshold_L, hidden_BL, decoder_norms_L)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.crosscoder.parameters()

    def before_backward_pass(self) -> None:
        self._synchronise_shared_latents_gradients()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
