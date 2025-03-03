from typing import TypeVar

import fire  # type: ignore

from model_diffing.data.token_hookpoint_dataloader import build_sliding_window_dataloader
from model_diffing.log import logger
from model_diffing.models.activations import JumpReLUActivation
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_jan_update_crosscoder.run import JanUpdateInitStrategy
from model_diffing.scripts.train_jumprelu_sliding_window.config import SlidingWindowExperimentConfig
from model_diffing.scripts.train_jumprelu_sliding_window.trainer import JumpReLUSlidingWindowCrosscoderTrainer
from model_diffing.scripts.train_l1_sliding_window.base_sliding_window_trainer import BiTokenCCWrapper
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import SaveableModule, get_device

TAct = TypeVar("TAct", bound=SaveableModule)


def _build_sliding_window_crosscoder_trainer(
    cfg: SlidingWindowExperimentConfig,
) -> JumpReLUSlidingWindowCrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )

    assert all("hook_resid" in hp for hp in cfg.hookpoints), "we should be training on the residual stream"

    dataloader = build_sliding_window_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
        window_size=2,
    )
    # assert next(dataloader.get_activations_iterator_BTPD()).shape[0] == cfg.train.minibatch_size(), "should be the same"

    crosscoder1, crosscoder2 = [
        AcausalCrosscoder(
            crosscoding_dims=(window_size, len(cfg.hookpoints)),
            d_model=llms[0].cfg.d_model,
            hidden_dim=cfg.crosscoder.hidden_dim,
            hidden_activation=JumpReLUActivation(
                size=cfg.crosscoder.hidden_dim,
                bandwidth=cfg.crosscoder.jumprelu.bandwidth,
                log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
                backprop_through_input=cfg.crosscoder.jumprelu.backprop_through_jumprelu_input,
            ),
            init_strategy=JanUpdateInitStrategy(
                activations_iterator_BXD=dataloader.get_shuffled_activations_iterator_BTPD(),
                initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
            ),
        )
        for window_size in [1, 2]
    ]

    crosscoders = BiTokenCCWrapper(crosscoder1, crosscoder2)
    crosscoders.to(device)

    wandb_run = build_wandb_run(cfg)

    return JumpReLUSlidingWindowCrosscoderTrainer(
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
    fire.Fire(run_exp(_build_sliding_window_crosscoder_trainer, SlidingWindowExperimentConfig))
