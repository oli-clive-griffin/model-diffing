from typing import Any

import fire  # type: ignore
import torch  # type: ignore
from einops import rearrange

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models import AcausalCrosscoder, AnthropicJumpReLUActivation
from model_diffing.models.acausal_crosscoder import InitStrategy
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.no_bias_jr.config import NoBiasJanUpdateExperimentConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import JanUpdateCrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device, random_direction_init_


class AnthropicTransposeInit(InitStrategy[AcausalCrosscoder[Any]]):
    def __init__(self, dec_init_norm: float, use_bias: bool):
        self.dec_init_norm = dec_init_norm
        self.use_bias = use_bias

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[Any]) -> None:
        random_direction_init_(cc.W_dec_HXD, self.dec_init_norm)

        cc.W_enc_XDH.copy_(rearrange(cc.W_dec_HXD.clone(), "h ... -> ... h"))

        cc.b_enc_H.zero_()
        cc.b_dec_XD.zero_()

        if not self.use_bias:
            cc.b_dec_XD.requires_grad_(False)
            cc.b_enc_H.requires_grad_(False)


def build_trainer(cfg: NoBiasJanUpdateExperimentConfig) -> JanUpdateCrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
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
        hidden_activation=AnthropicJumpReLUActivation(
            size=cfg.crosscoder.hidden_dim,
            bandwidth=cfg.crosscoder.jumprelu.bandwidth,
            log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
        ),
        init_strategy=AnthropicTransposeInit(dec_init_norm=0.1, use_bias=cfg.bias),
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


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(lambda cfg: build_trainer(cfg), NoBiasJanUpdateExperimentConfig))
