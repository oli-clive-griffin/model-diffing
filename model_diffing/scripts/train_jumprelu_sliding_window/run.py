from typing import Generic, TypeVar

import fire  # type: ignore
import torch
from einops import rearrange
from tqdm import tqdm  # type: ignore

from model_diffing.data.token_layer_dataloader import (
    BaseTokenLayerActivationsDataloader,
    build_sliding_window_dataloader,
)
from model_diffing.log import logger
from model_diffing.models.activations import JumpReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.train_jan_update_crosscoder.config import JumpReLUConfig
from model_diffing.scripts.train_jumprelu_sliding_window.config import SlidingWindowExperimentConfig
from model_diffing.scripts.train_jumprelu_sliding_window.trainer import (
    BiTokenCCWrapper,
    JumpreluSlidingWindowCrosscoderTrainer,
)
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import SaveableModule, get_device, inspect

TAct = TypeVar("TAct", bound=SaveableModule)


class TokenLayerCrosscoder(AcausalCrosscoder[TAct], Generic[TAct]):
    def __init__(
        self,
        token_window_size: int,
        n_layers: int,
        d_model: int,
        hidden_dim: int,
        dec_init_norm: float,
        hidden_activation: TAct,
    ):
        super().__init__(
            crosscoding_dims=(token_window_size, n_layers),
            d_model=d_model,
            hidden_dim=hidden_dim,
            dec_init_norm=dec_init_norm,
            hidden_activation=hidden_activation,
        )
        self.W_enc_TLDH = self.W_enc_XDH
        self.W_dec_HTLD = self.W_dec_HXD
        self.b_dec_TLD = self.b_dec_XD

        assert self.W_enc_TLDH.shape == (token_window_size, n_layers, d_model, hidden_dim)
        assert self.W_dec_HTLD.shape == (hidden_dim, token_window_size, n_layers, d_model)
        assert self.b_dec_TLD.shape == (token_window_size, n_layers, d_model)


def _build_sliding_window_crosscoder_trainer(
    cfg: SlidingWindowExperimentConfig,
) -> JumpreluSlidingWindowCrosscoderTrainer:
    device = get_device()

    dataloader = build_sliding_window_dataloader(cfg.data, cfg.train.batch_size, cfg.cache_dir, device)
    _, n_tokens, n_layers, d_model = dataloader.batch_shape_BTLD()
    assert n_tokens == 2

    if cfg.data.token_window_size != 2:
        raise ValueError(f"token_window_size must be 2, got {cfg.data.token_window_size}")

    crosscoder1, crosscoder2 = [
        _build_jumprelu_crosscoder(
            n_tokens=window_size,
            n_layers=n_layers,
            d_model=d_model,
            cc_hidden_dim=cfg.crosscoder.hidden_dim,
            jumprelu=cfg.crosscoder.jumprelu,
            dataloader=dataloader,
            device=device,
        )
        for window_size in [1, 2]
    ]

    crosscoders = BiTokenCCWrapper(crosscoder1, crosscoder2)
    crosscoders.to(device)

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return JumpreluSlidingWindowCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoders=crosscoders,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        experiment_name=cfg.experiment_name,
    )


def _build_jumprelu_crosscoder(
    n_tokens: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    jumprelu: JumpReLUConfig,
    dataloader: BaseTokenLayerActivationsDataloader,
    device: torch.device,  # for computing b_enc
) -> TokenLayerCrosscoder[JumpReLUActivation]:
    cc = TokenLayerCrosscoder(
        token_window_size=n_tokens,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=0,  # dec_init_norm doesn't matter here as we override weights below
        hidden_activation=JumpReLUActivation(
            size=cc_hidden_dim,
            bandwidth=jumprelu.bandwidth,
            threshold_init=jumprelu.threshold_init,
            backprop_through_input=jumprelu.backprop_through_jumprelu_input,
        ),
    )

    cc.to(device)

    with torch.no_grad():
        # parameters from the jan update doc

        n = float(n_tokens * n_layers * d_model)  # n is the size of the input space
        m = float(cc_hidden_dim)  # m is the size of the hidden space

        # W_dec ~ U(-1/n, 1/n) (from doc)
        cc.W_dec_HTLD.uniform_(-1.0 / n, 1.0 / n)

        # For now, assume we're in the X == Y case.
        # Therefore W_enc = (n/m) * W_dec^T
        cc.W_enc_TLDH.copy_(rearrange(cc.W_dec_HTLD, "h t l d -> t l d h") * (n / m))

        calibrated_b_enc_H = _compute_b_enc_H(
            dataloader=dataloader,
            W_enc_TLDH=cc.W_enc_TLDH,
            initial_jumprelu_threshold_H=cc.hidden_activation.log_threshold_H.exp(),
            device=device,
            n_examples_to_sample=500,  # this should be enough that the quantile is stable
            firing_sparsity=min(10_000 / m, 1),
        )
        cc.b_enc_H.copy_(calibrated_b_enc_H)

        # no data-dependent initialization of b_dec
        cc.b_dec_TLD.zero_()

    return cc


def _compute_b_enc_H(
    dataloader: BaseTokenLayerActivationsDataloader,
    W_enc_TLDH: torch.Tensor,
    initial_jumprelu_threshold_H: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int,
    firing_sparsity: float,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_examples_to_sample} examples")

    pre_bias_NH = _harvest_pre_bias_NH(dataloader, W_enc_TLDH, device, n_examples_to_sample)

    # find the threshold for each idx H such that "firing_sparsity" of the examples are above the threshold
    quantile_H = torch.quantile(pre_bias_NH, 1 - firing_sparsity, dim=0)

    # firing is when the post-bias is above the jumprelu threshold therefore, we subtract
    # the quantile from the initial jumprelu threshold, such the 1/firing_sparsity of the
    # examples are above the threshold
    b_enc_H = initial_jumprelu_threshold_H - quantile_H

    logger.info(f"computed b_enc_H. Sample: {b_enc_H[:10]}. mean: {b_enc_H.mean()}, std: {b_enc_H.std()}")

    return b_enc_H


def _harvest_pre_bias_NH(
    dataloader: BaseTokenLayerActivationsDataloader,
    W_enc_TLDH: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int,
) -> torch.Tensor:
    batch_size = dataloader.batch_shape_BTLD()[0]

    remainder = n_examples_to_sample % batch_size
    if remainder != 0:
        logger.warning(
            f"n_examples_to_sample {n_examples_to_sample} must be divisible by the batch "
            f"size {batch_size}. Rounding up to the nearest multiple of batch_size."
        )
        # Round up to the nearest multiple of batch_size:
        n_examples_to_sample = (((n_examples_to_sample - remainder) // batch_size) + 1) * batch_size

        logger.info(f"n_examples_to_sample is now {n_examples_to_sample}")

    num_batches = n_examples_to_sample // batch_size

    activations_iterator_BTLD = dataloader.get_shuffled_activations_iterator_BTLD()

    def get_batch_pre_bias_BH() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BTLD = next(activations_iterator_BTLD).to(device)
        return torch.einsum("b t l d, t l d h -> b h", batch_BTLD, W_enc_TLDH)

    first_sample_BH = get_batch_pre_bias_BH()
    hidden_size = first_sample_BH.shape[1]

    pre_bias_buffer_NH = torch.empty(n_examples_to_sample, hidden_size, device=device)
    logger.info(f"pre_bias_buffer_NH: {inspect(pre_bias_buffer_NH)}")

    pre_bias_buffer_NH[:batch_size] = first_sample_BH

    for i in tqdm(
        range(1, num_batches), desc="Harvesting pre-bias"
    ):  # start at 1 because we already sampled the first batch
        batch_pre_bias_BH = get_batch_pre_bias_BH()
        pre_bias_buffer_NH[batch_size * i : batch_size * (i + 1)] = batch_pre_bias_BH

    return pre_bias_buffer_NH


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(_build_sliding_window_crosscoder_trainer, SlidingWindowExperimentConfig))
