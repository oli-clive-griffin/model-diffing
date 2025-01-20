from pathlib import Path
from typing import cast

import fire
import torch
import wandb
import yaml
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.dataloader.data import ActivationHarvester, ShuffledTokensActivationsLoader
from model_diffing.models.crosscoder import build_l1_crosscoder
from model_diffing.scripts.train_l1_crosscoder.trainer import L1SaeTrainer
from model_diffing.utils import get_device

from .config import Config


def build_trainer(cfg: Config) -> L1SaeTrainer:
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

    crosscoder = build_l1_crosscoder(
        n_layers=len(cfg.layer_indices_to_harvest),
        d_model=llms[0].cfg.d_model,
        cc_hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=len(llms),
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

    expected_batch_shape = (
        cfg.train.batch_size,
        len(llms),
        len(cfg.layer_indices_to_harvest),
        llms[0].cfg.d_model,
    )

    return L1SaeTrainer(
        cfg=cfg.train,
        llms=llms,
        optimizer=optimizer,
        dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        expected_batch_shape=expected_batch_shape,
    )


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config


def main(config_path: str) -> None:
    print("Loading config...")
    config = load_config(Path(config_path))
    print("Loaded config")
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    print("Starting...")
    fire.Fire(main)
