from collections.abc import Iterator

import torch
from einops import rearrange

from crosscoding.log import logger
from crosscoding.models.activations import AnthropicSTEJumpReLUActivation
from crosscoding.models.initialization.init_strategy import InitStrategy
from crosscoding.models.initialization.jan_update_init import get_quantile_L, harvest_pre_bias_NL
from crosscoding.models.sparse_coders import ModelHookpointAcausalCrosscoder
from crosscoding.utils import random_direction_init_


class NoEncoderBiasJumpReLUInitStrategy(InitStrategy[ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation]]):
    def __init__(
        self,
        dec_init_norm: float,
        activations_iterator_BXD: Iterator[torch.Tensor],
        initial_approx_firing_pct: float,
        device: torch.device,
        n_tokens_for_threshold_setting: int,
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
    def init_weights(self, cc: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation]) -> None:
        # W
        random_direction_init_(cc.W_dec_LXD, self.dec_init_norm)
        cc.W_enc_XDL.copy_(rearrange(cc.W_dec_LXD.clone(), "l ... -> ... l"))

        # Bias
        assert cc.b_enc_L is None, "this strategy requires no encoder bias"

        if cc.b_dec_XD is not None:
            cc.b_dec_XD.zero_()

        # Jumprelu threshold
        jumprelu_threshold_L = compute_jumprelu_threshold_L(
            cc,
            self.activations_iterator_BXD,
            self.initial_approx_firing_pct,
            self.n_tokens_for_threshold_setting,
        )
        logger.info(f"computed jumprelu_threshold_L. sample: {jumprelu_threshold_L[:10]}")
        clamped_threshold_L = jumprelu_threshold_L.clamp(min=0.001)
        pct_clamped = (jumprelu_threshold_L < 0.001).float().mean().item()
        logger.info(f"clamped {pct_clamped:.2%} of jumprelu_threshold_L. sample: {clamped_threshold_L[:10]}")
        cc.activation_fn.log_threshold_L.copy_(clamped_threshold_L.log())


def compute_jumprelu_threshold_L(
    cc: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation],
    activations_iterator_BXD: Iterator[torch.Tensor],
    initial_approx_firing_pct: float,
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_tokens_for_threshold_setting} examples")
    pre_bias_NL = harvest_pre_bias_NL(cc, activations_iterator_BXD, n_tokens_for_threshold_setting)

    logger.info("computing pre-bias firing threshold quantile")
    return get_quantile_L(pre_bias_NL, initial_approx_firing_pct)
