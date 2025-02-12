from typing import TypeVar

import fire  # type: ignore
import torch
from einops import rearrange

from model_diffing.data.token_hookpoint_dataloader import (
    BaseTokenhookpointActivationsDataloader,
    build_sliding_window_dataloader,
)
from model_diffing.log import logger
from model_diffing.models.activations import JumpReLUActivation
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_jan_update_crosscoder.config import JumpReLUConfig
from model_diffing.scripts.train_jan_update_crosscoder.run import compute_b_enc_H
from model_diffing.scripts.train_jumprelu_sliding_window.config import SlidingWindowExperimentConfig
from model_diffing.scripts.train_jumprelu_sliding_window.trainer import JumpreluSlidingWindowCrosscoderTrainer
from model_diffing.scripts.train_l1_sliding_window.trainer import BiTokenCCWrapper, TokenhookpointCrosscoder
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import SaveableModule, get_device

TAct = TypeVar("TAct", bound=SaveableModule)


def _build_sliding_window_crosscoder_trainer(
    cfg: SlidingWindowExperimentConfig,
) -> JumpreluSlidingWindowCrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )

    assert all("hook_resid" in hp for hp in cfg.hookpoints), "we're assuming we're training on the residual stream"

    dataloader = build_sliding_window_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.batch_size,
        cache_dir=cfg.cache_dir,
        device=device,
    )

    if cfg.data.token_window_size != 2:
        raise ValueError(f"token_window_size must be 2, got {cfg.data.token_window_size}")

    crosscoder1, crosscoder2 = [
        _build_jumprelu_crosscoder(
            n_tokens=window_size,
            n_hookpoints=len(cfg.hookpoints),
            d_model=llms[0].cfg.d_model,
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
        hookpoints=cfg.hookpoints,
        experiment_name=cfg.experiment_name,
    )


def _build_jumprelu_crosscoder(
    n_tokens: int,
    n_hookpoints: int,
    d_model: int,
    cc_hidden_dim: int,
    jumprelu: JumpReLUConfig,
    dataloader: BaseTokenhookpointActivationsDataloader,
    device: torch.device,  # for computing b_enc
) -> TokenhookpointCrosscoder[JumpReLUActivation]:
    cc = TokenhookpointCrosscoder(
        token_window_size=n_tokens,
        n_hookpoints=n_hookpoints,
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

        n = float(n_tokens * n_hookpoints * d_model)  # n is the size of the input space
        m = float(cc_hidden_dim)  # m is the size of the hidden space

        # W_dec ~ U(-1/n, 1/n) (from doc)
        cc.W_dec_HTPD.uniform_(-1.0 / n, 1.0 / n)

        # For now, assume we're in the X == Y case.
        # Therefore W_enc = (n/m) * W_dec^T
        cc.W_enc_TPDH.copy_(rearrange(cc.W_dec_HTPD, "h t p d -> t p d h") * (n / m))

        calibrated_b_enc_H = compute_b_enc_H(
            activations_iterator_BXD=dataloader.get_shuffled_activations_iterator_BTPD(),
            W_enc_XDH=cc.W_enc_TPDH,
            initial_jumprelu_threshold_H=cc.hidden_activation.log_threshold_H.exp(),
            device=device,
            n_examples_to_sample=500,  # this should be enough that the quantile is stable
            firing_sparsity=min(10_000 / m, 1),
        )
        cc.b_enc_H.copy_(calibrated_b_enc_H)

        # no data-dependent initialization of b_dec
        cc.b_dec_TPD.zero_()

    return cc


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(_build_sliding_window_crosscoder_trainer, SlidingWindowExperimentConfig))
