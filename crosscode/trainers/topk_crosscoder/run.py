import fire  # type: ignore

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.log import logger
from crosscode.models import AnthropicTransposeInit, ModelHookpointAcausalCrosscoder, TopkActivation
from crosscode.models.activations.topk import BatchTopkActivation, GroupMaxActivation
from crosscode.trainers.base_trainer import run_exp
from crosscode.trainers.topk_crosscoder.config import TopKAcausalCrosscoderExperimentConfig
from crosscode.trainers.topk_crosscoder.trainer import TopKStyleAcausalCrosscoderTrainer
from crosscode.trainers.utils import build_wandb_run
from crosscode.utils import get_device


def build_trainer(cfg: TopKAcausalCrosscoderExperimentConfig) -> TopKStyleAcausalCrosscoderTrainer:
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

    return TopKStyleAcausalCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        model=crosscoder,
        wandb_run=wandb_run,
        device=device,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopKAcausalCrosscoderExperimentConfig))
