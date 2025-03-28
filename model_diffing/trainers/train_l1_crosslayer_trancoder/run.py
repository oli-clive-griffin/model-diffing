import fire  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models import ReLUActivation
from model_diffing.models.crosscoder import CrossLayerTranscoder
from model_diffing.models.initialization.anthropic_transpose import AnthropicTransposeInitCrossLayerTC
from model_diffing.trainers.base_trainer import run_exp
from model_diffing.trainers.llms import build_llms
from model_diffing.trainers.train_l1_crosslayer_trancoder.config import L1CrossLayerTranscoderExperimentConfig
from model_diffing.trainers.train_l1_crosslayer_trancoder.trainer import L1CrossLayerTranscoderTrainer
from model_diffing.trainers.utils import build_wandb_run
from model_diffing.utils import get_device


def build_l1_crosscoder_trainer(cfg: L1CrossLayerTranscoderExperimentConfig) -> L1CrossLayerTranscoderTrainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 1, "the trainer assumes we have one model"
    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    d_model = llms[0].cfg.d_model
    n_layers_out = len(cfg.hookpoints) - 1

    crosscoder = CrossLayerTranscoder(
        d_model=d_model,
        n_layers_out=n_layers_out,
        n_latents=cfg.crosscoder.n_latents,
        activation_fn=ReLUActivation(),
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
        init_strategy=AnthropicTransposeInitCrossLayerTC(dec_init_norm=cfg.crosscoder.dec_init_norm),
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return L1CrossLayerTranscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_l1_crosscoder_trainer, L1CrossLayerTranscoderExperimentConfig))
