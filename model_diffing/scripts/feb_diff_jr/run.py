from typing import TypeVar

import fire  # type: ignore
import torch  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.models.diffing_crosscoder import DiffingCrosscoder, ModelDiffingDataDependentJumpReLUInitStrategy
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateExperimentConfig
from model_diffing.scripts.feb_diff_jr.trainer import ModelDiffingFebUpdateJumpReLUTrainer
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import SaveableModule, get_device

TActivation = TypeVar("TActivation", bound=SaveableModule)


def build_feb_update_crosscoder_trainer(
    cfg: JumpReLUModelDiffingFebUpdateExperimentConfig,
) -> ModelDiffingFebUpdateJumpReLUTrainer:
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
        hookpoints=[cfg.hookpoint],
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    crosscoder = DiffingCrosscoder(
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        n_explicitly_shared_latents=cfg.crosscoder.n_shared_latents,
        init_strategy=ModelDiffingDataDependentJumpReLUInitStrategy(
            activations_iterator_BXD=dataloader.get_activations_iterator_BMPD(),
            initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
            n_tokens_for_threshold_setting=cfg.crosscoder.n_tokens_for_threshold_setting,
            device=device,
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

    return ModelDiffingFebUpdateJumpReLUTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=[cfg.hookpoint],
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_feb_update_crosscoder_trainer, JumpReLUModelDiffingFebUpdateExperimentConfig))
