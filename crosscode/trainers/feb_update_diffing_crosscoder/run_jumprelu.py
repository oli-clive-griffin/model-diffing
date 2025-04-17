import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import (
    AnthropicSTEJumpReLUActivation,
    DataDependentJumpReLUInitStrategy,
    ModelHookpointAcausalCrosscoder,
)
from crosscode.models.initialization.diffing_identical_latents import IdenticalLatentsInit
from crosscode.trainers.feb_update_diffing_crosscoder.config import JumpReLUModelDiffingFebUpdateExperimentConfig
from crosscode.trainers.feb_update_diffing_crosscoder.jumprelu_trainer import JumpReLUModelDiffingFebUpdateWrapper
from crosscode.trainers.trainer import Trainer, run_exp
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_feb_update_crosscoder_trainer(cfg: JumpReLUModelDiffingFebUpdateExperimentConfig) -> Trainer:
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

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=len(llms),
        n_hookpoints=1,
        d_model=llms[0].cfg.d_model,
        n_latents=cfg.crosscoder.n_latents,
        init_strategy=IdenticalLatentsInit(
            first_init=DataDependentJumpReLUInitStrategy(
                activations_iterator=dataloader.get_activations_iterator(),
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

    wrapper = JumpReLUModelDiffingFebUpdateWrapper(
        model=crosscoder,
        n_shared_latents=cfg.crosscoder.n_shared_latents,
        hookpoints=[cfg.hookpoint],
        model_names=[llm.name or "unknown" for llm in llms],
        save_dir=cfg.save_dir,
        scaling_factors_MP=dataloader.get_scaling_factors(),
        lambda_p=cfg.train.lambda_p,
        num_steps=cfg.train.num_steps,
        final_lambda_s=cfg.train.final_lambda_s,
        final_lambda_f=cfg.train.final_lambda_f,
        c=cfg.train.c,
    )

    wandb_run = build_wandb_run(cfg)

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
    fire.Fire(run_exp(build_feb_update_crosscoder_trainer, JumpReLUModelDiffingFebUpdateExperimentConfig))
