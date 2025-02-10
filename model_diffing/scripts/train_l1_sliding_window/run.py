import fire  # type: ignore

from model_diffing.data.token_layer_dataloader import build_sliding_window_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.train_jumprelu_sliding_window.run import TokenLayerCrosscoder
from model_diffing.scripts.train_l1_sliding_window.config import L1SlidingWindowExperimentConfig
from model_diffing.scripts.train_l1_sliding_window.trainer import BiTokenCCWrapper, L1SlidingWindowCrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device


def _build_sliding_window_crosscoder_trainer(cfg: L1SlidingWindowExperimentConfig) -> L1SlidingWindowCrosscoderTrainer:
    device = get_device()

    dataloader = build_sliding_window_dataloader(cfg.data, cfg.train.batch_size, cfg.cache_dir, device)
    _, n_tokens, n_layers, d_model = dataloader.batch_shape_BTLD()
    assert n_tokens == 2

    if cfg.data.token_window_size != 2:
        raise ValueError(f"token_window_size must be 2, got {cfg.data.token_window_size}")

    crosscoder1, crosscoder2 = [
        TokenLayerCrosscoder(
            token_window_size=window_size,
            n_layers=n_layers,
            d_model=d_model,
            hidden_dim=cfg.crosscoder.hidden_dim,
            dec_init_norm=cfg.crosscoder.dec_init_norm,
            hidden_activation=ReLUActivation(),
        )
        for window_size in [1, 2]
    ]

    crosscoders = BiTokenCCWrapper(crosscoder1, crosscoder2)
    crosscoders.to(device)

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return L1SlidingWindowCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoders=crosscoders,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        experiment_name=cfg.experiment_name,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(_build_sliding_window_crosscoder_trainer, L1SlidingWindowExperimentConfig))
