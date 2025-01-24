from pathlib import Path

import fire
import yaml

from model_diffing.dataloader.data import build_dataloader_BMLD
from model_diffing.log import logger
from model_diffing.models.crosscoder import build_relu_crosscoder
from model_diffing.scripts.train_l1_crosscoder.config import L1ExperimentConfig
from model_diffing.scripts.train_l1_crosscoder.trainer import L1CrosscoderTrainer
from model_diffing.utils import build_wandb_run, get_device


def build_l1_crosscoder_trainer(cfg: L1ExperimentConfig) -> L1CrosscoderTrainer:
    device = get_device()

    dataloader_BMLD, (_, n_layers, n_models, d_model) = build_dataloader_BMLD(cfg.data, cfg.cache_dir, device)

    crosscoder = build_relu_crosscoder(
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=n_models,
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg.wandb, cfg) if cfg.wandb != "disabled" else None

    return L1CrosscoderTrainer(
        cfg=cfg.train,
        dataloader_BMLD=dataloader_BMLD,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
    )


def load_config(config_path: Path) -> L1ExperimentConfig:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = L1ExperimentConfig(**config_dict)
    return config


def main(config_path: str) -> None:
    logger.info("Loading config...")
    config = load_config(Path(config_path))
    logger.info("Loaded config")
    trainer = build_l1_crosscoder_trainer(config)
    trainer.train()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
