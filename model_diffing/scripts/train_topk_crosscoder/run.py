from pathlib import Path

import fire
import torch
import yaml

from model_diffing.dataloader.data import build_dataloader_BMLD
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_topk_crosscoder.config import TopKExperimentConfig
from model_diffing.scripts.train_topk_crosscoder.trainer import TopKTrainer
from model_diffing.utils import build_wandb_run, get_device


def build_trainer(cfg: TopKExperimentConfig) -> TopKTrainer:
    device = get_device()

    llms = build_llms(cfg.llms, cfg.cache_dir, device)

    dataloader_BMLD = build_dataloader_BMLD(cfg.data, llms, cfg.cache_dir)

    crosscoder = AcausalCrosscoder(
        n_layers=len(cfg.data.activations_iterator.layer_indices_to_harvest),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=len(llms),
        hidden_activation=torch.relu,
    )
    crosscoder = crosscoder.to(device)

    initial_lr = cfg.train.learning_rate.initial_learning_rate
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=initial_lr)

    wandb_run = build_wandb_run(cfg.wandb)

    return TopKTrainer(
        cfg=cfg.train,
        llms=llms,
        optimizer=optimizer,
        dataloader_BMLD=dataloader_BMLD,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
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
