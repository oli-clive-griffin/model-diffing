from pathlib import Path
from typing import Any

import torch

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.activations.jumprelu import AnthropicSTEJumpReLUActivation
from crosscode.trainers.crosscoder_wrapper import CrosscoderWrapper
from crosscode.trainers.utils import get_l0_stats, wandb_histogram
from crosscode.utils import (
    calculate_reconstruction_loss_summed_norm_MSEs,
    get_summed_decoder_norms_L,
    not_none,
    pre_act_loss,
    tanh_sparsity_loss,
)


class JanUpdateModelHookpointAcausalCrosscoderWrapper(CrosscoderWrapper[AnthropicSTEJumpReLUActivation]):
    def __init__(
        self,
        model: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation],
        scaling_factors_MP: torch.Tensor,
        lambda_p: float,
        hookpoints: list[str],
        model_names: list[str],
        save_dir: Path,
        num_steps: int,
        final_lambda_s: float,
        c: float,
    ):
        super().__init__(
            model,
            scaling_factors_MP,
            hookpoints,
            model_names,
            save_dir,
        )

        self.lambda_p = lambda_p
        self.num_steps = num_steps
        self.final_lambda_s = final_lambda_s
        self.c = c

    def _calculate_loss_and_log(
        self,
        step: int,
        batch: ModelHookpointActivationsBatch,
        train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(
            batch.activations_BMPD, train_res.recon_acts_BMPD
        )
        decoder_norms_L = get_summed_decoder_norms_L(self.crosscoder.W_dec_LMPD)
        tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.latents_BL, decoder_norms_L)
        pre_act_loss = self._pre_act_loss(train_res.latents_BL, decoder_norms_L)

        lambda_s = self._lambda_s_scheduler(step)

        loss = (
            reconstruction_loss  #
            + lambda_s * tanh_sparsity_loss
            + self.lambda_p * pre_act_loss
        )

        if log:
            log_dict: dict[str, float] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/pre_act_loss": pre_act_loss.item(),
                "train/loss": loss.item(),
                "train/lambda_s": lambda_s,
                **self._get_fvu_dict(batch.activations_BMPD, train_res.recon_acts_BMPD),
                **get_l0_stats(train_res.latents_BL),
            }

            return loss, log_dict

        return loss, None

    def expensive_logs(self) -> dict[str, Any]:
        return {
            **super().expensive_logs(),
            "media/jr_threshold": wandb_histogram(self.crosscoder.activation_fn.log_threshold_L.exp()),
            "media/jr_threshold_grad": wandb_histogram(not_none(self.crosscoder.activation_fn.log_threshold_L.grad)),
        }

    def _lambda_s_scheduler(self, step: int) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (step / self.num_steps) * self.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BL: torch.Tensor, decoder_norms_L: torch.Tensor) -> torch.Tensor:
        return tanh_sparsity_loss(self.c, hidden_BL, decoder_norms_L)

    def _pre_act_loss(self, hidden_BL: torch.Tensor, decoder_norms_L: torch.Tensor) -> torch.Tensor:
        return pre_act_loss(self.crosscoder.activation_fn.log_threshold_L, hidden_BL, decoder_norms_L)
