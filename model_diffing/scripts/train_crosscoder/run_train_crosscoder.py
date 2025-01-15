"""Train a model on MNIST.

This script takes ~40 seconds to run for 3 layers and 15 epochs on a CPU.

Usage:
    python model_diffing/scripts/train_mnist/run_train_mnist.py <path/to/config.yaml>
"""

from functools import partial
from pathlib import Path
from typing import cast

import fire
import torch
import wandb
import yaml
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_crosscoder.data import (
    ActivationHarvester,
    ShuffledActivationLoader,
)
from model_diffing.utils import save_model_and_config

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
CACHE_DIR = "tmp/cache"


class TrainConfig(BaseModel):
    lr: float = 5e-5
    lambda_max: float = 5.0
    lambda_n_steps: int = 1000
    batch_size: int
    epochs: int
    save_dir: Path | None
    save_every_n_epochs: int | None


class CrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1


class DatasetConfig(BaseModel):
    hf_dataset: str
    cache_dir: str
    sequence_length: int
    harvest_batch_size: int
    shuffle_buffer_size: int


class WandbConfig(BaseModel):
    project: str
    entity: str


class Config(BaseModel):
    seed: int
    model_names: list[str]
    layer_indices_to_harvest: list[int]
    train: TrainConfig
    crosscoder: CrosscoderConfig
    dataset: DatasetConfig
    wandb: WandbConfig | None
    dtype: torch.dtype = torch.float32


def train(cfg: Config):
    llms: list[HookedTransformer] = [
        cast(
            HookedTransformer,
            HookedTransformer.from_pretrained(name, cache_dir=CACHE_DIR, dtype=str(cfg.dtype)).to(
                DEVICE
            ),
        )
        for name in cfg.model_names
    ]
    assert len({llm.cfg.d_model for llm in llms}) == 1, "All models must have the same d_model"
    d_model = llms[0].cfg.d_model

    assert all(llm.tokenizer == llms[0].tokenizer for llm in llms), (
        "All models must have the same tokenizer"
    )
    tokenizer = llms[0].tokenizer
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    crosscoder = AcausalCrosscoder(
        n_layers=len(cfg.layer_indices_to_harvest),
        d_model=d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=len(llms),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=cfg.train.lr)

    activation_harvester = ActivationHarvester(
        hf_dataset=cfg.dataset.hf_dataset,
        cache_dir=CACHE_DIR,
        models=llms,
        tokenizer=tokenizer,
        sequence_length=cfg.dataset.sequence_length,
        batch_size=cfg.dataset.harvest_batch_size,
        layer_indices_to_harvest=cfg.layer_indices_to_harvest,
    )

    dataloader = ShuffledActivationLoader(
        activation_harvester=activation_harvester,
        shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
        batch_size=cfg.train.batch_size,
    )

    lambda_step = partial(
        lambda_scheduler,
        lambda_max=cfg.train.lambda_max,
        n_steps=cfg.train.lambda_n_steps,
    )

    for step, batch_Ns_LD in enumerate(dataloader):
        batch_Ns_LD = batch_Ns_LD.to(DEVICE)
        optimizer.zero_grad()

        _, losses = crosscoder.forward_train(batch_Ns_LD)

        lambda_ = lambda_step(step=step)
        loss = losses.reconstruction_loss + lambda_ * losses.sparsity_loss

        print(
            f"Step {step:05d}, loss: {loss.item():.4f}, "
            f"lambda: {lambda_:.4f}, "
            f"reconstruction_loss: {losses.reconstruction_loss.item():.4f}, "
            f"sparsity_loss: {losses.sparsity_loss.item():.4f}"
        )

        if cfg.wandb:
            wandb.log({"train/loss": loss.item(), "train/lambda": lambda_})

        clip_grad_norm_(crosscoder.parameters(), 1.0)
        loss.backward()
        optimizer.step()

        if (
            cfg.train.save_dir
            and cfg.train.save_every_n_epochs
            and (step + 1) % cfg.train.save_every_n_epochs == 0
        ):
            save_model_and_config(
                config=cfg, save_dir=cfg.train.save_dir, model=crosscoder, epoch=step
            )


def lambda_scheduler(lambda_max: float, n_steps: int, step: int):
    if step < n_steps:
        return lambda_max * step / n_steps
    else:
        return lambda_max


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
