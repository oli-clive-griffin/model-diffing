import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models.activations.topk import TopkActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.models.initialization.anthropic_transpose import AnthropicTransposeInitCrossLayerTC
from crosscode.trainers.topk_cross_layer_transcoder.config import TopkCrossLayerTranscoderExperimentConfig
from crosscode.trainers.topk_cross_layer_transcoder.trainer import TopkCrossLayerTranscoderWrapper
from crosscode.trainers.trainer import Trainer, run_exp
from crosscode.trainers.utils import build_wandb_run, get_activation_type
from crosscode.utils import get_device


def build_trainer(cfg: TopkCrossLayerTranscoderExperimentConfig) -> Trainer:
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
        hookpoints=[cfg.in_hookpoint, *cfg.out_hookpoints],
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    activation_type = get_activation_type([cfg.in_hookpoint, *cfg.out_hookpoints])
    act_dim = llms[0].cfg.d_mlp if activation_type == "mlp" else llms[0].cfg.d_model

    transcoder = CrossLayerTranscoder(
        d_model=act_dim,
        n_layers_out=len(cfg.out_hookpoints),
        n_latents=cfg.transcoder.n_latents,
        linear_skip=cfg.transcoder.linear_skip,
        init_strategy=AnthropicTransposeInitCrossLayerTC(dec_init_norm=cfg.transcoder.dec_init_norm),
        activation_fn=TopkActivation(k=cfg.transcoder.k),
        use_encoder_bias=cfg.transcoder.use_encoder_bias,
        use_decoder_bias=cfg.transcoder.use_decoder_bias,
    )

    transcoder = transcoder.to(device)

    wandb_run = build_wandb_run(cfg)

    if cfg.train.k_aux is None:
        cfg.train.k_aux = act_dim // 2
        logger.info(f"defaulting to k_aux={cfg.train.k_aux} for crosscoder (({act_dim=}) // 2)")

    model_wrapper = TopkCrossLayerTranscoderWrapper(
        model=transcoder,
        save_dir=cfg.save_dir,
        scaling_factors_P=dataloader.get_scaling_factors(),
        lambda_aux=cfg.train.lambda_aux,
        k_aux=cfg.train.k_aux,
        dead_latents_threshold_n_examples=cfg.train.dead_latents_threshold_n_examples,
        hookpoints_out=cfg.out_hookpoints,
    )

    return Trainer(
        activations_dataloader=dataloader,
        model=model_wrapper,
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
    fire.Fire(run_exp(build_trainer, TopkCrossLayerTranscoderExperimentConfig))
