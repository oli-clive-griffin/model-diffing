import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models.activations.topk import TopkActivation
from crosscode.models.crosslayer_transcoder import CrossLayerTranscoder
from crosscode.models.initialization.transcoder import ZeroDecSkipTranscoderInit
from crosscode.trainers.base_trainer import run_exp
from crosscode.trainers.skip_transcoder.config import TopkSkipTranscoderExperimentConfig
from crosscode.trainers.skip_transcoder.trainer import TopkSkipTransCrosscoderTrainer
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_trainer(cfg: TopkSkipTranscoderExperimentConfig) -> TopkSkipTransCrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    in_hookpoint = f"blocks.{cfg.mlp_index_in}.mlp.hook_pre"
    out_hookpoints = [f"blocks.{idx}.mlp.hook_post" for idx in cfg.mlp_indices_out]

    harvesting_hookpoints = [in_hookpoint, *out_hookpoints]

    dataloader = build_model_hookpoint_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=harvesting_hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    d_mlp = llms[0].cfg.d_mlp  # this doesn't seem right?

    transcoder = CrossLayerTranscoder(
        d_model=d_mlp,
        n_layers_out=len(cfg.mlp_indices_out),
        n_latents=cfg.transcoder.n_latents,
        linear_skip=cfg.transcoder.linear_skip,
        init_strategy=ZeroDecSkipTranscoderInit(
            activation_iterator=dataloader.get_activations_iterator(),
            n_samples_for_dec_mean=100_000,
            enc_init_norm=0.1,  # cfg.crosscoder.enc_init_norm,
        ),
        activation_fn=TopkActivation(k=cfg.transcoder.k),
        use_encoder_bias=cfg.transcoder.use_encoder_bias,
        use_decoder_bias=cfg.transcoder.use_decoder_bias,
    )

    transcoder = transcoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return TopkSkipTransCrosscoderTrainer(
        cfg=cfg.train,
        out_hookpoints=out_hookpoints,
        activations_dataloader=dataloader,
        model=transcoder,
        wandb_run=wandb_run,
        device=device,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopkSkipTranscoderExperimentConfig))
