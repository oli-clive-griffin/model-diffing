from collections.abc import Iterator
from math import prod

import torch
from einops import rearrange

from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.models.activations import AnthropicJumpReLUActivation
from model_diffing.models.initialization.jan_update_init import harvest_pre_bias_NH
from model_diffing.utils import random_direction_init_


class NoBiasJumpReLUInitStrategy(InitStrategy[AcausalCrosscoder[AnthropicJumpReLUActivation]]):
    def __init__(
        self,
        dec_init_norm: float,
        activations_iterator_BXD: Iterator[torch.Tensor],
        initial_approx_firing_pct: float,
        device: torch.device,
    ):
        """
        Args:
            activations_iterator_BXD: iterator over activations with which to calibrate the initial jumprelu threshold
            initial_approx_firing_pct: percentage of examples that should fire. In the update, this value is 10_000/n\
                But we're often training with n << 10_000, so we allow setting this value directly.
        """
        self.activations_iterator_BXD = activations_iterator_BXD
        self.n_tokens_for_threshold_setting = n_tokens_for_threshold_setting
        self.device = device

        if initial_approx_firing_pct < 0 or initial_approx_firing_pct > 1:
            raise ValueError(f"initial_approx_firing_pct must be between 0 and 1, got {initial_approx_firing_pct}")

        self.initial_approx_firing_pct = initial_approx_firing_pct

        self.dec_init_norm = dec_init_norm



    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[AnthropicJumpReLUActivation]) -> None:
        # W
        random_direction_init_(cc.W_dec_HXD, self.dec_init_norm)
        cc.W_enc_XDH.copy_(rearrange(cc.W_dec_HXD.clone(), "h ... -> ... h"))

        # Bias
        cc.b_enc_H.zero_()
        cc.b_dec_XD.zero_()
        # cc.b_enc_H.requires_grad_(False)
        # cc.b_dec_XD.requires_grad_(False)

        # JR
        jumprelu_threshold_H = compute_jumprelu_threshold_H(
            self.activations_iterator_BXD,
            cc.W_enc_XDH,
            self.initial_approx_firing_pct,
            self.n_tokens_for_threshold_setting,
        )
        logger.info(f"computed jumprelu_threshold_H. sample: {jumprelu_threshold_H[:10]}")
        clamped_threshold_H = jumprelu_threshold_H.clamp(min=0.001)
        pct_clamped = (jumprelu_threshold_H < 0.001).float().mean().item()
        logger.info(f"clamped {pct_clamped:.2%} of jumprelu_threshold_H. sample: {clamped_threshold_H[:10]}")
        cc.hidden_activation.log_threshold_H.copy_(clamped_threshold_H.log())



def compute_jumprelu_threshold_H(
    activations_iterator_BXD: Iterator[torch.Tensor],
    W_enc_XDH: torch.Tensor,
    initial_approx_firing_pct: float,
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_tokens_for_threshold_setting} examples")
    pre_bias_NH = harvest_pre_bias_NH(W_enc_XDH, activations_iterator_BXD, n_tokens_for_threshold_setting)

    logger.info("computing pre-bias firing threshold quantile")
    
    return get_quantile_H(pre_bias_NH, initial_approx_firing_pct)


def get_quantile_H(pre_bias_NH: torch.Tensor, initial_approx_firing_pct: float) -> torch.Tensor:
    pre_bias_firing_threshold_quantile_H = torch.empty(pre_bias_NH.shape[1])
    n_chunks = pre_bias_firing_threshold_quantile_H.shape[0] // 1024
    for i in range(n_chunks):
        start, end = i * 1024, (i + 1) * 1024
        pre_bias_firing_threshold_quantile_H[start:end] = torch.quantile(
            pre_bias_NH[:, start:end],
            1 - initial_approx_firing_pct,
            dim=0,
        )

    return pre_bias_firing_threshold_quantile_H
