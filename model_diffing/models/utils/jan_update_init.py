from collections.abc import Iterator

import torch
from einops import einsum
from tqdm import tqdm  # type: ignore

from model_diffing.log import logger
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.utils import inspect, round_up


class BaseDataDependentJumpReLUInitStrategy:
    """
    Base class for the data-dependent JumpReLU initialization strategy described in:
        https://transformer-circuits.pub/2025/january-update/index.html.
    """

    def __init__(
        self,
        activations_iterator_BXD: Iterator[torch.Tensor],
        initial_approx_firing_pct: float,
        device: torch.device,
        n_tokens_for_threshold_setting: int = 100_000,
    ):
        """
        Args:
            activations_iterator_BXD: iterator over activations with which to calibrate the initial jumprelu threshold
            initial_approx_firing_pct: percentage of examples that should fire. In the update, this value is 10_000/n\
                But we're often training with n << 10_000, so we allow setting this value directly.
            n_tokens_for_threshold_setting: number of examples to sample
        """
        self.activations_iterator_BXD = activations_iterator_BXD
        self.n_tokens_for_threshold_setting = n_tokens_for_threshold_setting
        self.device = device

        if initial_approx_firing_pct < 0 or initial_approx_firing_pct > 1:
            raise ValueError(f"initial_approx_firing_pct must be between 0 and 1, got {initial_approx_firing_pct}")
        self.initial_approx_firing_pct = initial_approx_firing_pct

    def get_calibrated_b_enc_H(
        self, W_enc_XDH: torch.Tensor, hidden_activation: AnthropicJumpReLUActivation
    ) -> torch.Tensor:
        return compute_b_enc_H(
            self.activations_iterator_BXD,
            W_enc_XDH.to(self.device),
            hidden_activation.log_threshold_H.exp(),
            self.initial_approx_firing_pct,
            self.n_tokens_for_threshold_setting,
        )


def compute_b_enc_H(
    activations_iterator_BXD: Iterator[torch.Tensor],
    W_enc_XDH: torch.Tensor,
    initial_jumprelu_threshold_H: torch.Tensor,
    initial_approx_firing_pct: float,
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_tokens_for_threshold_setting} examples")

    pre_bias_NH = _harvest_pre_bias_NH(W_enc_XDH, activations_iterator_BXD, n_tokens_for_threshold_setting)

    logger.info("computing pre-bias firing threshold quantile")

    pre_bias_firing_threshold_quantile_H = torch.empty_like(initial_jumprelu_threshold_H)
    n_chunks = pre_bias_firing_threshold_quantile_H.shape[0] // 1024
    for i in range(n_chunks):
        start, end = i * 1024, (i + 1) * 1024
        pre_bias_firing_threshold_quantile_H[start:end] = torch.quantile(
            pre_bias_NH[:, start:end],
            1 - initial_approx_firing_pct,
            dim=0,
        )

    # firing is when the post-bias is above the jumprelu threshold, therefore we subtract
    # the quantile from the initial jumprelu threshold, so that for a given example,
    # inital_approx_firing_pct of the examples are above the threshold.
    b_enc_H = initial_jumprelu_threshold_H - pre_bias_firing_threshold_quantile_H.to(
        initial_jumprelu_threshold_H.device
    )

    logger.info(f"computed b_enc_H. Sample: {b_enc_H[:10]}. mean: {b_enc_H.mean()}, std: {b_enc_H.std()}")

    return b_enc_H


def _harvest_pre_bias_NH(
    W_enc_XDH: torch.Tensor,
    activations_iterator_BXD: Iterator[torch.Tensor],
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    # print(f"W_enc_XDH.device: {W_enc_XDH.device}")
    # print(f"activations_iterator_BXD.device: {next(activations_iterator_BXD).device}")
    # print(f"device: {device}")

    def get_batch_pre_bias() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BXD = next(activations_iterator_BXD)
        # TODO make this a method
        x_BH = einsum(batch_BXD.to(W_enc_XDH.device), W_enc_XDH, "b ... d, ... d h -> b h")
        return x_BH

    sample_BH = get_batch_pre_bias()
    batch_size, hidden_size = sample_BH.shape

    rounded_n_tokens_for_threshold_setting = round_up(n_tokens_for_threshold_setting, to_multiple_of=batch_size)

    if rounded_n_tokens_for_threshold_setting > n_tokens_for_threshold_setting:
        logger.warning(
            f"rounded n_tokens_for_threshold_setting from {n_tokens_for_threshold_setting} "
            f"to {rounded_n_tokens_for_threshold_setting} to be divisible by the batch size {batch_size}"
        )

    num_batches = rounded_n_tokens_for_threshold_setting // batch_size

    pre_bias_buffer_NH = torch.empty(
        (rounded_n_tokens_for_threshold_setting, hidden_size),
        device=W_enc_XDH.device,
    )

    logger.info(f"pre_bias_buffer_NH: {inspect(pre_bias_buffer_NH)}")

    pre_bias_buffer_NH[:batch_size] = sample_BH

    # start at 1 because we already sampled the first batch
    for i in tqdm(range(1, num_batches), desc="Harvesting pre-bias"):
        batch_pre_bias_BH = get_batch_pre_bias()
        pre_bias_buffer_NH[i * batch_size : (i + 1) * batch_size] = batch_pre_bias_BH

    print(f"pre_bias_buffer_NH.device: {pre_bias_buffer_NH.device}")
    return pre_bias_buffer_NH
