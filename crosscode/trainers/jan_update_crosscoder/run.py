import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import (
    AnthropicSTEJumpReLUActivation,
    DataDependentJumpReLUInitStrategy,
    ModelHookpointAcausalCrosscoder,
)
from crosscode.trainers.base_trainer import run_exp
from crosscode.trainers.jan_update_crosscoder.config import JanUpdateExperimentConfig
from crosscode.trainers.jan_update_crosscoder.trainer import JanUpdateModelHookpointAcausalCrosscoderTrainer
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_jan_update_crosscoder_trainer(
    cfg: JanUpdateExperimentConfig,
) -> JanUpdateModelHookpointAcausalCrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=len(llms),
        n_hookpoints=len(cfg.hookpoints),
        d_model=llms[0].cfg.d_model,
        n_latents=cfg.crosscoder.n_latents,
        init_strategy=DataDependentJumpReLUInitStrategy(
            activations_iterator=dataloader.get_activations_iterator(),
            initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
            n_tokens_for_threshold_setting=cfg.crosscoder.n_tokens_for_threshold_setting,
            device=device,
        ),
        activation_fn=AnthropicSTEJumpReLUActivation(
            size=cfg.crosscoder.n_latents,
            bandwidth=cfg.crosscoder.jumprelu.bandwidth,
            log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
        ),
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return JanUpdateModelHookpointAcausalCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        model=crosscoder,
        wandb_run=wandb_run,
        device=device,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_jan_update_crosscoder_trainer, JanUpdateExperimentConfig))
