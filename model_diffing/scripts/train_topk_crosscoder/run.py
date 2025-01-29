from pathlib import Path

import fire  # type: ignore
import yaml  # type: ignore

from model_diffing.dataloader.data import build_dataloader
from model_diffing.log import logger
from model_diffing.models.crosscoder import build_topk_crosscoder
from model_diffing.scripts.train_topk_crosscoder.config import TopKExperimentConfig
from model_diffing.scripts.train_topk_crosscoder.trainer import TopKTrainer
from model_diffing.utils import build_wandb_run, get_device


def build_trainer(cfg: TopKExperimentConfig) -> TopKTrainer:
    device = get_device()

    dataloader = build_dataloader(cfg.data, cfg.cache_dir, device)
    _, n_layers, n_models, d_model = dataloader.batch_shape_BMLD()

    crosscoder = build_topk_crosscoder(
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=n_models,
        k=cfg.crosscoder.k,
    )
    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return TopKTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        experiment_name=cfg.experiment_name,
    )


def load_config(config_path: Path) -> TopKExperimentConfig:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = TopKExperimentConfig(**config_dict)
    return config


def main(config_path: str) -> None:
    logger.info("Loading config...")
    config = load_config(Path(config_path))
    logger.info("Loaded config")
    logger.info(f"Training with {config.model_dump_json()}")
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
