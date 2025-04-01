from collections.abc import Iterator

import torch
from einops import rearrange
from tqdm import tqdm  # type: ignore

from crosscoding.log import logger
from crosscoding.models.activations import AnthropicSTEJumpReLUActivation
from crosscoding.models.initialization.init_strategy import InitStrategy
from crosscoding.models.sparse_coders import ModelHookpointAcausalCrosscoder
from crosscoding.utils import ceil_div, inspect, round_up


class DataDependentJumpReLUInitStrategy(InitStrategy[ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation]]):
    """
    Base class for the data-dependent JumpReLU initialization strategy described in:
        https://transformer-circuits.pub/2025/january-update/index.html.
    """

    def __init__(
        self,
        activations_iterator_BMPD: Iterator[torch.Tensor],
        initial_approx_firing_pct: float,
        device: torch.device,
        n_tokens_for_threshold_setting: int = 100_000,
    ):
        """
        Args:
            activations_iterator_BMPD: iterator over activations with which to calibrate the initial jumprelu threshold
            initial_approx_firing_pct: percentage of examples that should fire. In the update, this value is 10_000/n\
                But we're often training with n << 10_000, so we allow setting this value directly.
            n_tokens_for_threshold_setting: number of examples to sample
        """
        self.activations_iterator_BMPD = activations_iterator_BMPD
        self.n_tokens_for_threshold_setting = n_tokens_for_threshold_setting
        self.device = device

        if initial_approx_firing_pct < 0 or initial_approx_firing_pct > 1:
            raise ValueError(f"initial_approx_firing_pct must be between 0 and 1, got {initial_approx_firing_pct}")
        self.initial_approx_firing_pct = initial_approx_firing_pct

    @torch.no_grad()
    def init_weights(self, cc: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation]) -> None:
        n = cc.n_models * cc.n_hookpoints * cc.d_model
        m = cc.n_latents

        cc.W_dec_LMPD.uniform_(-1.0 / n, 1.0 / n)
        cc.W_enc_MPDL.copy_(
            rearrange(cc.W_dec_LMPD, "l ... -> ... l")  #
            * (n / m)
        )

        assert cc.b_enc_L is not None, "this strategy requires an encoder bias"
        cc.b_enc_L.copy_(self._get_calibrated_b_enc_L(cc, cc.activation_fn).to(cc.b_enc_L.device))

        if cc.b_dec_MPD is not None:
            cc.b_dec_MPD.zero_()

    def _get_calibrated_b_enc_L(
        self,
        cc: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation],
        hidden_activation: AnthropicSTEJumpReLUActivation,
    ) -> torch.Tensor:
        return compute_b_enc_L(
            cc,
            self.activations_iterator_BMPD,
            hidden_activation.log_threshold_L.exp(),
            self.initial_approx_firing_pct,
            self.n_tokens_for_threshold_setting,
        )


def compute_b_enc_L(
    cc: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation],
    activations_iterator_BMPD: Iterator[torch.Tensor],
    initial_jumprelu_threshold_L: torch.Tensor,
    initial_approx_firing_pct: float,
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_tokens_for_threshold_setting} examples")

    pre_bias_NL = harvest_pre_bias_NL(cc, activations_iterator_BMPD, n_tokens_for_threshold_setting)

    pre_bias_firing_threshold_quantile_L = get_quantile_L(pre_bias_NL, initial_approx_firing_pct)

    # firing is when the post-bias is above the jumprelu threshold, therefore we subtract
    # the quantile from the initial jumprelu threshold, so that for a given example,
    # inital_approx_firing_pct of the examples are above the threshold.
    b_enc_L = initial_jumprelu_threshold_L - pre_bias_firing_threshold_quantile_L.to(
        initial_jumprelu_threshold_L.device
    )

    logger.info(f"computed b_enc_L. Sample: {b_enc_L[:10]}. mean: {b_enc_L.mean()}, std: {b_enc_L.std()}")

    return b_enc_L


def harvest_pre_bias_NL(
    cc: ModelHookpointAcausalCrosscoder[AnthropicSTEJumpReLUActivation],
    activations_iterator_BMPD: Iterator[torch.Tensor],
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    sample_BL = cc.get_pre_bias_BL(next(activations_iterator_BMPD))
    batch_size, latent_size = sample_BL.shape

    rounded_n_tokens_for_threshold_setting = round_up(n_tokens_for_threshold_setting, to_multiple_of=batch_size)

    if rounded_n_tokens_for_threshold_setting > n_tokens_for_threshold_setting:
        logger.warning(
            f"rounded n_tokens_for_threshold_setting from {n_tokens_for_threshold_setting} "
            f"to {rounded_n_tokens_for_threshold_setting} to be divisible by the batch size {batch_size}"
        )

    num_batches = rounded_n_tokens_for_threshold_setting // batch_size

    pre_bias_buffer_NL = torch.empty(
        (rounded_n_tokens_for_threshold_setting, latent_size),
        device=cc.W_enc_MPDL.device,
    )

    logger.info(f"pre_bias_buffer_NL: {inspect(pre_bias_buffer_NL)}")

    pre_bias_buffer_NL[:batch_size] = sample_BL

    # start at 1 because we already sampled the first batch
    for i in tqdm(range(1, num_batches), desc="Harvesting pre-bias"):
        batch_pre_bias_BL = cc.get_pre_bias_BL(next(activations_iterator_BMPD))
        pre_bias_buffer_NL[i * batch_size : (i + 1) * batch_size] = batch_pre_bias_BL

    return pre_bias_buffer_NL


CHUNK_SIZE = 4096


def get_quantile_L(pre_bias_NL: torch.Tensor, initial_approx_firing_pct: float) -> torch.Tensor:
    pre_bias_firing_threshold_quantile_L = torch.empty(pre_bias_NL.shape[1])
    n_chunks = ceil_div(pre_bias_firing_threshold_quantile_L.shape[0], CHUNK_SIZE)
    for i in tqdm(range(n_chunks), desc="computing pre-bias firing threshold quantile"):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, pre_bias_NL.shape[1])
        pre_bias_firing_threshold_quantile_L[start:end] = torch.quantile(
            pre_bias_NL[:, start:end],
            1 - initial_approx_firing_pct,
            dim=0,
        )

    return pre_bias_firing_threshold_quantile_L
