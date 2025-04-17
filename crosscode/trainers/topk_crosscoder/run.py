import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import AnthropicTransposeInit, ModelHookpointAcausalCrosscoder, TopkActivation
from crosscode.models.activations.topk import BatchTopkActivation, GroupMaxActivation
from crosscode.trainers.topk_crosscoder.config import TopKAcausalCrosscoderExperimentConfig
from crosscode.trainers.topk_crosscoder.trainer import TopKAcausalCrosscoderWrapper
from crosscode.trainers.trainer import Trainer, run_exp
from crosscode.trainers.utils import build_optimizer, build_wandb_run
from crosscode.utils import get_device


def build_trainer(cfg: TopKAcausalCrosscoderExperimentConfig) -> Trainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    match cfg.train.topk_style:
        case "topk":
            cc_act = TopkActivation(k=cfg.crosscoder.k)
        case "batch_topk":
            cc_act = BatchTopkActivation(k_per_example=cfg.crosscoder.k)
        case "groupmax":
            cc_act = GroupMaxActivation(k_groups=cfg.crosscoder.k, latents_size=cfg.crosscoder.n_latents)

    d_model = llms[0].cfg.d_model

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=len(llms),
        n_hookpoints=len(cfg.hookpoints),
        d_model=d_model,
        n_latents=cfg.crosscoder.n_latents,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
        activation_fn=cc_act,
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
    )

    crosscoder = crosscoder.to(device)

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    if cfg.train.k_aux is None:
        cfg.train.k_aux = d_model // 2
        logger.info(f"defaulting to k_aux={cfg.train.k_aux} for crosscoder (({d_model=}) // 2)")

    wandb_run = build_wandb_run(cfg)

    optimizer = build_optimizer(cfg.train.optimizer, params=crosscoder.parameters())
    lr_scheduler = None # build_lr_scheduler(cfg.train.optimizer, num_steps=cfg.train.num_steps)

    model_wrapper = TopKAcausalCrosscoderWrapper(
        model=crosscoder,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
        scaling_factors_MP=dataloader.get_scaling_factors(),
        lambda_aux=cfg.train.lambda_aux,
        k_aux=cfg.train.k_aux,
        dead_latents_threshold_n_examples=cfg.train.dead_latents_threshold_n_examples,
    )

    return Trainer(
        num_steps=cfg.train.num_steps,
        gradient_accumulation_steps_per_batch=cfg.train.gradient_accumulation_steps_per_batch,
        save_every_n_steps=cfg.train.save_every_n_steps,
        log_every_n_steps=cfg.train.log_every_n_steps,
        upload_saves_to_wandb=cfg.train.upload_saves_to_wandb,
        activations_dataloader=dataloader,
        model_wrapper=model_wrapper,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopKAcausalCrosscoderExperimentConfig))
