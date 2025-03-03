import fire  # type: ignore

from model_diffing.data.token_hookpoint_dataloader import build_sliding_window_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_l1_crosscoder.trainer import AnthropicTransposeInit
from model_diffing.scripts.train_l1_sliding_window.base_sliding_window_trainer import BiTokenCCWrapper
from model_diffing.scripts.train_l1_sliding_window.config import L1SlidingWindowExperimentConfig
from model_diffing.scripts.train_l1_sliding_window.trainer import L1SlidingWindowCrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device


def _build_sliding_window_crosscoder_trainer(cfg: L1SlidingWindowExperimentConfig) -> L1SlidingWindowCrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )

    assert len({llm.cfg.d_model for llm in llms}) == 1, "all models must have the same d_model"
    d_model = llms[0].cfg.d_model

    
    dataloader = build_sliding_window_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
        window_size=2,
    )

    crosscoder1, crosscoder2 = [
        AcausalCrosscoder(
            crosscoding_dims=(window_size, len(cfg.hookpoints)),
            d_model=d_model,
            hidden_dim=cfg.crosscoder.hidden_dim,
            init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
            hidden_activation=ReLUActivation(),
        )
        for window_size in [1, 2]
    ]

    crosscoders = BiTokenCCWrapper(crosscoder1, crosscoder2)
    crosscoders.to(device)

    wandb_run = build_wandb_run(cfg)

    return L1SlidingWindowCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoders=crosscoders,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(_build_sliding_window_crosscoder_trainer, L1SlidingWindowExperimentConfig))
