from pathlib import Path
from typing import cast

import fire
import torch
import wandb
import yaml
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.dataloader.activations import ActivationHarvester, ShuffledTokensActivationsLoader
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_topk_crosscoder.config import TopKConfig
from model_diffing.scripts.train_topk_crosscoder.trainer import TopKTrainer
from model_diffing.utils import get_device


def build_trainer(cfg: TopKConfig) -> TopKTrainer:
    device = get_device()

    llms = [
        cast(
            HookedTransformer,  # for some reason, the type checker thinks this is simply an nn.Module
            HookedTransformer.from_pretrained(
                model.name,
                revision=model.revision,
                cache_dir=cfg.dataset.cache_dir,
                dtype=str(cfg.dtype),
            ).to(device),
        )
        for model in cfg.llms
    ]
    tokenizer = llms[0].tokenizer
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    tokenizer = tokenizer

    dataloader = ShuffledTokensActivationsLoader(
        activation_harvester=ActivationHarvester(
            hf_dataset=cfg.dataset.hf_dataset,
            cache_dir=cfg.dataset.cache_dir,
            models=llms,
            tokenizer=tokenizer,
            sequence_length=cfg.dataset.sequence_length,
            batch_size=cfg.dataset.harvest_batch_size,
            layer_indices_to_harvest=cfg.layer_indices_to_harvest,
        ),
        shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
        batch_size=cfg.train.batch_size,
    )

    crosscoder = AcausalCrosscoder(
        n_layers=len(cfg.layer_indices_to_harvest),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=len(llms),
        hidden_activation=torch.relu,
    )
    crosscoder = crosscoder.to(device)

    initial_lr = cfg.train.learning_rate.initial_learning_rate
    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=initial_lr)

    wandb_run = (
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=cfg.model_dump(),
        )
        if cfg.wandb
        else None
    )

    return TopKTrainer(
        cfg=cfg.train,
        llms=llms,
        optimizer=optimizer,
        dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
    )


def load_config(config_path: Path) -> TopKConfig:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = TopKConfig(**config_dict)
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
