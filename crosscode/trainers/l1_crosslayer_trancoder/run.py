import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import ReLUActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.models.initialization.anthropic_transpose import AnthropicTransposeInitCrossLayerTC
from crosscode.trainers.l1_crosslayer_trancoder.config import L1CrossLayerTranscoderExperimentConfig
from crosscode.trainers.l1_crosslayer_trancoder.trainer import L1CrossLayerTranscoderWrapper
from crosscode.trainers.trainer import Trainer, run_exp
from crosscode.trainers.utils import build_wandb_run, get_activation_type
from crosscode.utils import get_device


def build_l1_cross_layer_transcoder_trainer(cfg: L1CrossLayerTranscoderExperimentConfig) -> Trainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 1, "the trainer assumes we have one model"
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

    crosscoder = CrossLayerTranscoder(
        d_model=act_dim,
        n_layers_out=len(cfg.out_hookpoints),
        n_latents=cfg.transcoder.n_latents,
        activation_fn=ReLUActivation(),
        use_encoder_bias=cfg.transcoder.use_encoder_bias,
        use_decoder_bias=cfg.transcoder.use_decoder_bias,
        init_strategy=AnthropicTransposeInitCrossLayerTC(dec_init_norm=cfg.transcoder.dec_init_norm),
    )

    model_wrapper = L1CrossLayerTranscoderWrapper(
        model=crosscoder,
        scaling_factors_P=dataloader.get_scaling_factors(),
        hookpoints_out=cfg.out_hookpoints,
        save_dir=cfg.save_dir,
        lambda_s_num_steps=cfg.train.lambda_s_num_steps,
        final_lambda_s=cfg.train.final_lambda_s,
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

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
    fire.Fire(run_exp(build_l1_cross_layer_transcoder_trainer, L1CrossLayerTranscoderExperimentConfig))
