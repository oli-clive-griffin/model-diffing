import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import AnthropicTransposeInit, ReLUActivation
from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.trainers.l1_crosscoder.config import L1ExperimentConfig
from crosscode.trainers.l1_crosscoder.trainer import L1AcausalCrosscoderWrapper
from crosscode.trainers.trainer import Trainer, run_exp
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_l1_crosscoder_trainer(cfg: L1ExperimentConfig) -> Trainer:
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
        activation_fn=ReLUActivation(),
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
    )

    model = L1AcausalCrosscoderWrapper(
        model=crosscoder.to(device),
        scaling_factors_MP=dataloader.get_scaling_factors(),
        hookpoints=cfg.hookpoints,
        model_names=[llm.name or "" for llm in llms],  # fixme
        save_dir=cfg.save_dir,
        lambda_s_num_steps=cfg.train.lambda_s_num_steps,
        final_lambda_s=cfg.train.final_lambda_s,
    )

    wandb_run = build_wandb_run(cfg)

    return Trainer(
        activations_dataloader=dataloader,
        model=model,
        optimizer_cfg=cfg.train.optimizer,
        wandb_run=wandb_run,

        # make this into a "train loop cfg"?
        num_steps=cfg.train.num_steps,
        gradient_accumulation_microbatches_per_step=cfg.train.gradient_accumulation_microbatches_per_step,
        log_every_n_steps=cfg.train.log_every_n_steps,
        save_every_n_steps=cfg.train.save_every_n_steps,
        upload_saves_to_wandb=cfg.train.upload_saves_to_wandb,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_l1_crosscoder_trainer, L1ExperimentConfig))
