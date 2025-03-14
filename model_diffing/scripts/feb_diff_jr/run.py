import fire  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models import AcausalCrosscoder, AnthropicJumpReLUActivation, DataDependentJumpReLUInitStrategy
from model_diffing.scripts.base_diffing_trainer import IdenticalLatentsInit
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateExperimentConfig
from model_diffing.scripts.feb_diff_jr.trainer import ModelDiffingFebUpdateJumpReLUTrainer
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device


def build_feb_update_crosscoder_trainer(
    cfg: JumpReLUModelDiffingFebUpdateExperimentConfig,
) -> ModelDiffingFebUpdateJumpReLUTrainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 2, "expected 2 models for model-diffing"

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        inferenced_type=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=[cfg.hookpoint],
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(2,),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        init_strategy=IdenticalLatentsInit(
            first_init=DataDependentJumpReLUInitStrategy(
                activations_iterator_BXD=dataloader.get_activations_iterator_BMPD(),
                initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
                n_tokens_for_threshold_setting=cfg.crosscoder.n_tokens_for_threshold_setting,
                device=device,
            ),
            n_shared_latents=cfg.crosscoder.n_shared_latents,
        ),
        hidden_activation=AnthropicJumpReLUActivation(
            size=cfg.crosscoder.hidden_dim,
            bandwidth=cfg.crosscoder.jumprelu.bandwidth,
            log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
        ),
    )

    return ModelDiffingFebUpdateJumpReLUTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder.to(device),
        n_shared_latents=cfg.crosscoder.n_shared_latents,
        wandb_run=build_wandb_run(cfg),
        device=device,
        hookpoints=[cfg.hookpoint],
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_feb_update_crosscoder_trainer, JumpReLUModelDiffingFebUpdateExperimentConfig))
