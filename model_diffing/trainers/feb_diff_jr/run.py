from collections import OrderedDict

import fire  # type: ignore

from model_diffing.data.base_activations_dataloader import CrosscodingDim, CrosscodingDims
from model_diffing.data.model_hookpoint_dataloader import build_model_hookpoint_dataloader
from model_diffing.log import logger
from model_diffing.models import AcausalCrosscoder, AnthropicSTEJumpReLUActivation, DataDependentJumpReLUInitStrategy
from model_diffing.trainers.base_diffing_trainer import IdenticalLatentsInit
from model_diffing.trainers.base_trainer import run_exp
from model_diffing.trainers.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateExperimentConfig
from model_diffing.trainers.feb_diff_jr.trainer import ModelDiffingFebUpdateJumpReLUTrainer
from model_diffing.trainers.llms import build_llms
from model_diffing.trainers.utils import build_wandb_run
from model_diffing.utils import get_device


def build_feb_update_crosscoder_trainer(
    cfg: JumpReLUModelDiffingFebUpdateExperimentConfig,
) -> ModelDiffingFebUpdateJumpReLUTrainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 2, "expected 2 models for model-diffing"

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=[cfg.hookpoint],
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    crosscoding_dims = CrosscodingDims(
        [
            ("model", CrosscodingDim(name="model", index_labels=["0", "1"])),
            ("hookpoint", CrosscodingDim(name="hookpoint", index_labels=[cfg.hookpoint])),
        ]
    )

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=crosscoding_dims,
        d_model=llms[0].cfg.d_model,
        n_latents=cfg.crosscoder.n_latents,
        init_strategy=IdenticalLatentsInit(
            first_init=DataDependentJumpReLUInitStrategy(
                activations_iterator_BXD=dataloader.get_activations_iterator_BXD(),
                initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
                n_tokens_for_threshold_setting=cfg.crosscoder.n_tokens_for_threshold_setting,
                device=device,
            ),
            n_shared_latents=cfg.crosscoder.n_shared_latents,
        ),
        activation_fn=AnthropicSTEJumpReLUActivation(
            size=cfg.crosscoder.n_latents,
            bandwidth=cfg.crosscoder.jumprelu.bandwidth,
            log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
        ),
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
    )

    return ModelDiffingFebUpdateJumpReLUTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder.to(device),
        wandb_run=build_wandb_run(cfg),
        device=device,
        save_dir=cfg.save_dir,
        n_shared_latents=cfg.crosscoder.n_shared_latents,
        crosscoding_dims=crosscoding_dims,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_feb_update_crosscoder_trainer, JumpReLUModelDiffingFebUpdateExperimentConfig))
