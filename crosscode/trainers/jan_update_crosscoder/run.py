import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import (
    AnthropicSTEJumpReLUActivation,
    DataDependentJumpReLUInitStrategy,
    ModelHookpointAcausalCrosscoder,
)
from crosscode.trainers.jan_update_crosscoder.config import JanUpdateExperimentConfig
from crosscode.trainers.jan_update_crosscoder.trainer import JanUpdateModelHookpointAcausalCrosscoderWrapper
from crosscode.trainers.trainer import Trainer, run_exp
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_jan_update_crosscoder_trainer(cfg: JanUpdateExperimentConfig) -> Trainer:
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

    wrapper = JanUpdateModelHookpointAcausalCrosscoderWrapper(
        model=crosscoder,
        scaling_factors_MP=dataloader.get_scaling_factors(),
        lambda_p=cfg.train.lambda_p,
        hookpoints=cfg.hookpoints,
        model_names=[llm.name or "unknown" for llm in llms],
        save_dir=cfg.save_dir,
        num_steps=cfg.train.num_steps,
        final_lambda_s=cfg.train.final_lambda_s,
        c=cfg.train.c,
    )

    return Trainer(
        activations_dataloader=dataloader,
        model=wrapper,
        optimizer_cfg=cfg.train.optimizer,
        wandb_run=wandb_run,

        # make this into a "train loop cfg"?
        num_steps=cfg.train.num_steps,
        gradient_accumulation_microbatches_per_step=cfg.train.gradient_accumulation_microbatches_per_step,
        save_every_n_steps=cfg.train.save_every_n_steps,
        log_every_n_steps=cfg.train.log_every_n_steps,
        upload_saves_to_wandb=cfg.train.upload_saves_to_wandb,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_jan_update_crosscoder_trainer, JanUpdateExperimentConfig))
