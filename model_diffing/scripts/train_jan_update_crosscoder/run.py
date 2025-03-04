from collections.abc import Iterator
from math import prod

import fire  # type: ignore
import torch
from einops import einsum, rearrange
from tqdm import tqdm  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.models.activations import AnthropicJumpReLUActivation
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateExperimentConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import JanUpdateCrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device, inspect, round_up


def build_jan_update_crosscoder_trainer(cfg: JanUpdateExperimentConfig) -> JanUpdateCrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    n_models = len(llms)
    n_hookpoints = len(cfg.hookpoints)

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        init_strategy=DataDependentJumpReLUInitStrategy(
            activations_iterator_BXD=dataloader.get_activations_iterator_BMPD(),
            initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
            n_tokens_for_threshold_setting=cfg.crosscoder.n_tokens_for_threshold_setting,
        ),
        hidden_activation=AnthropicJumpReLUActivation(
            size=cfg.crosscoder.hidden_dim,
            bandwidth=cfg.crosscoder.jumprelu.bandwidth,
            log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
            backprop_through_input=cfg.crosscoder.jumprelu.backprop_through_jumprelu_input,
        ),
    )
    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return JanUpdateCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )


class DataDependentJumpReLUInitStrategy(InitStrategy[AcausalCrosscoder[AnthropicJumpReLUActivation]]):
    def __init__(
        self,
        activations_iterator_BXD: Iterator[torch.Tensor],
        initial_approx_firing_pct: float,
        n_tokens_for_threshold_setting: int = 100_000,
    ):
        """
        Args:
            activations_iterator_BXD: iterator over activations with which to calibrate the initial jumprelu threshold
            initial_approx_firing_pct: percentage of examples that should fire. In the update, this value is 10_000 / n. But we're often training with n << 10_000, so we allow setting this value directly.
            n_tokens_for_threshold_setting: number of examples to sample
        """
        self.activations_iterator_BXD = activations_iterator_BXD
        self.n_tokens_for_threshold_setting = n_tokens_for_threshold_setting

        if initial_approx_firing_pct < 0 or initial_approx_firing_pct > 1:
            raise ValueError(f"initial_approx_firing_pct must be between 0 and 1, got {initial_approx_firing_pct}")
        self.initial_approx_firing_pct = initial_approx_firing_pct

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[AnthropicJumpReLUActivation]) -> None:
        n = prod(cc.crosscoding_dims) * cc.d_model
        m = cc.hidden_dim

        cc.W_dec_HXD.uniform_(-1.0 / n, 1.0 / n)
        cc.W_enc_XDH.copy_(
            rearrange(cc.W_dec_HXD, "hidden ... -> ... hidden")  #
            * (n / m)
        )

        calibrated_b_enc_H = compute_b_enc_H(
            self.activations_iterator_BXD,
            cc.W_enc_XDH,
            cc.hidden_activation.log_threshold_H.exp(),
            self.initial_approx_firing_pct,
            self.n_tokens_for_threshold_setting,
        )

        cc.b_enc_H.copy_(calibrated_b_enc_H)

        cc.b_dec_XD.zero_()


def compute_b_enc_H(
    activations_iterator_BXD: Iterator[torch.Tensor],
    W_enc_XDH: torch.Tensor,
    initial_jumprelu_threshold_H: torch.Tensor,
    initial_approx_firing_pct: float,
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_tokens_for_threshold_setting} examples")

    # pre-bias is just x @ W
    pre_bias_NH = _harvest_pre_bias_NH(W_enc_XDH, activations_iterator_BXD, n_tokens_for_threshold_setting)

    pre_bias_firing_threshold_quantile_H = torch.quantile(
        pre_bias_NH,
        1 - initial_approx_firing_pct,
        dim=0,
    )

    # firing is when the post-bias is above the jumprelu threshold, therefore we subtract
    # the quantile from the initial jumprelu threshold, so that for a given example,
    # inital_approx_firing_pct of the examples are above the threshold.
    b_enc_H = initial_jumprelu_threshold_H - pre_bias_firing_threshold_quantile_H

    logger.info(f"computed b_enc_H. Sample: {b_enc_H[:10]}. mean: {b_enc_H.mean()}, std: {b_enc_H.std()}")

    return b_enc_H


def _harvest_pre_bias_NH(
    W_enc_XDH: torch.Tensor,
    activations_iterator_BXD: Iterator[torch.Tensor],
    n_tokens_for_threshold_setting: int,
) -> torch.Tensor:
    cc_device = W_enc_XDH.device

    def get_batch_pre_bias() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BXD = next(activations_iterator_BXD).to(cc_device)
        x_BH = einsum(batch_BXD, W_enc_XDH, "b ... d, ... d h -> b h")
        return x_BH

    sample_BH = get_batch_pre_bias()
    batch_size, hidden_size = sample_BH.shape

    rounded_n_tokens_for_threshold_setting = round_up(n_tokens_for_threshold_setting, to_multiple_of=batch_size)

    if rounded_n_tokens_for_threshold_setting > n_tokens_for_threshold_setting:
        logger.warning(
            f"rounded n_tokens_for_threshold_setting from {n_tokens_for_threshold_setting} to {rounded_n_tokens_for_threshold_setting} "
            f"to be divisible by the batch size {batch_size}"
        )

    num_batches = rounded_n_tokens_for_threshold_setting // batch_size

    pre_bias_buffer_NH = torch.empty(rounded_n_tokens_for_threshold_setting, hidden_size, device=cc_device)
    logger.info(f"pre_bias_buffer_NH: {inspect(pre_bias_buffer_NH)}")

    pre_bias_buffer_NH[:batch_size] = sample_BH

    # start at 1 because we already sampled the first batch
    for i in tqdm(range(1, num_batches), desc="Harvesting pre-bias"):
        batch_pre_bias_BH = get_batch_pre_bias()
        pre_bias_buffer_NH[i * batch_size : (i + 1) * batch_size] = batch_pre_bias_BH

    return pre_bias_buffer_NH


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_jan_update_crosscoder_trainer, JanUpdateExperimentConfig))
