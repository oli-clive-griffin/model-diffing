import fire  # type: ignore

from crosscoding.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscoding.llms import build_llms
from crosscoding.log import logger
from crosscoding.models import AnthropicTransposeInit, ModelHookpointAcausalCrosscoder, TopkActivation
from crosscoding.models.activations.topk import BatchTopkActivation, GroupMaxActivation
from crosscoding.trainers.base_trainer import run_exp
from crosscoding.trainers.train_topk_crosscoder.config import TopKExperimentConfig
from crosscoding.trainers.train_topk_crosscoder.trainer import TopKStyleTrainer
from crosscoding.trainers.utils import build_wandb_run
from crosscoding.utils import get_device


def build_trainer(cfg: TopKExperimentConfig) -> TopKStyleTrainer:
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
        hookpoints=cfg.hookpoints,
        d_model=d_model,
        n_latents=cfg.crosscoder.n_latents,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
        activation_fn=cc_act,
        use_encoder_bias=cfg.crosscoder.use_encoder_bias,
        use_decoder_bias=cfg.crosscoder.use_decoder_bias,
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

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

    return TopKStyleTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopKExperimentConfig))
