from typing import Any, TypeVar

import fire  # type: ignore
import torch
from einops import rearrange

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models import InitStrategy
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.diffing_crosscoder import DiffingCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.feb_diff_l1.config import L1ModelDiffingFebUpdateExperimentConfig
from model_diffing.scripts.feb_diff_l1.trainer import ModelDiffingFebUpdateL1Trainer
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import SaveableModule, get_device, random_direction_init_

TActivation = TypeVar("TActivation", bound=SaveableModule)


def build_feb_update_crosscoder_trainer(cfg: L1ModelDiffingFebUpdateExperimentConfig) -> ModelDiffingFebUpdateL1Trainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 2, "we only support 2 models for now"

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

    crosscoder = DiffingCrosscoder(
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        n_explicitly_shared_latents=cfg.crosscoder.n_shared_latents,
        init_strategy=ModelDiffingAnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
        hidden_activation=ReLUActivation(size=cfg.crosscoder.hidden_dim),
    )
    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return ModelDiffingFebUpdateL1Trainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )


class ModelDiffingAnthropicTransposeInit(InitStrategy[DiffingCrosscoder[Any]]):
    """
    link: https://transformer-circuits.pub/2024/april-update/index.html#training-saes:~:text=are%20initialized%20to,in%20most%20cases

    to make encoder vectors point in random directions, we use normal_
    """

    def __init__(self, dec_init_norm: float):
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: DiffingCrosscoder[Any]) -> None:
        # First, initialize the decoder weights to point in random directions, and have
        random_direction_init_(cc._W_dec_shared_m0_HsD, self.dec_init_norm)
        random_direction_init_(cc._W_dec_indep_HiMD, self.dec_init_norm)

        cc.W_enc_MDH.copy_(rearrange(cc.theoretical_decoder_W_dec_HMD().clone(), "h ... -> ... h"))

        cc.b_enc_H.zero_()
        cc.b_dec_MD.zero_()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_feb_update_crosscoder_trainer, L1ModelDiffingFebUpdateExperimentConfig))
