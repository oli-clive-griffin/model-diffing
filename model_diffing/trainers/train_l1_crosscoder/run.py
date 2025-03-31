import fire  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import build_model_hookpoint_dataloader
from model_diffing.log import logger
from model_diffing.models import AnthropicTransposeInit, ReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.trainers.base_trainer import run_exp
from model_diffing.trainers.llms import build_llms
from model_diffing.trainers.train_l1_crosscoder.config import L1ExperimentConfig
from model_diffing.trainers.train_l1_crosscoder.trainer import L1CrosscoderTrainer
from model_diffing.trainers.utils import build_wandb_run
from model_diffing.utils import get_device


def build_l1_crosscoder_trainer(cfg: L1ExperimentConfig) -> L1CrosscoderTrainer:
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

    crosscoding_dims = [
        ("model", list(map(str, range(len(llms))))),
        ("hookpoint", cfg.hookpoints),
    ]

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=tuple(len(dim_labels) for _, dim_labels in crosscoding_dims),
        d_model=llms[0].cfg.d_model,
        n_latents=cfg.crosscoder.n_latents,
        activation_fn=ReLUActivation(),
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return L1CrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        save_dir=cfg.save_dir,
        crosscoding_dims=crosscoding_dims,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_l1_crosscoder_trainer, L1ExperimentConfig))
