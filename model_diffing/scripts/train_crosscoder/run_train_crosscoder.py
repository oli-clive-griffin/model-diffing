"""Train a model on MNIST.

This script takes ~40 seconds to run for 3 layers and 15 epochs on a CPU.

Usage:
    python model_diffing/scripts/train_mnist/run_train_mnist.py <path/to/config.yaml>
"""

from collections.abc import Iterator
from functools import partial
from itertools import islice
from pathlib import Path
from typing import cast

import fire
import numpy as np
import torch
import wandb
import yaml
from einops import reduce
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_crosscoder.data import (
    ActivationHarvester,
    ShuffledActivationLoader,
)
from model_diffing.utils import l2_norm, save_model_and_config

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class TrainConfig(BaseModel):
    lr: float = 5e-5
    lambda_max: float = 5.0
    lambda_n_steps: int = 1000
    batch_size: int
    epochs: int
    save_dir: Path | None
    save_every_n_epochs: int | None
    log_every_n_steps: int


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


class ModelConfig(BaseModel):
    name: str
    revision: str | None


class Config(BaseModel):
    seed: int
    models: list[ModelConfig]
    layer_indices_to_harvest: list[int]
    train: TrainConfig
    crosscoder: CrosscoderConfig
    dataset: DatasetConfig
    wandb: WandbConfig | None
    dtype: str = "float32"


@torch.no_grad()
def estimate_norm_scaling_factor(
    activations_dataloader: Iterator[torch.Tensor],
    d_model: int,
    n_batches_for_norm_estimate: int = 100,
) -> torch.Tensor:
    # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
    norms_per_batch = []
    for batch_BMLD in tqdm(
        islice(activations_dataloader, n_batches_for_norm_estimate),
        desc="Estimating norm scaling factor",
    ):
        norms_BML = reduce(batch_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
        norms_mean = norms_BML.mean().item()
        norms_per_batch.append(norms_mean)
    mean_norm = np.mean(norms_per_batch)
    scaling_factor = np.sqrt(d_model) / mean_norm
    return scaling_factor


def train(cfg: Config):
    llms: list[HookedTransformer] = [
        cast(
            HookedTransformer,
            HookedTransformer.from_pretrained(
                model.name,
                revision=model.revision,
                cache_dir=cfg.dataset.cache_dir,
                dtype=str(cfg.dtype),
            ).to(DEVICE),
        )
        for model in cfg.models
    ]
    assert len({llm.cfg.d_model for llm in llms}) == 1, "All models must have the same d_model"
    d_model = llms[0].cfg.d_model

    # assert all(llm.tokenizer == llms[0].tokenizer for llm in llms), (
    #     "All models must have the same tokenizer"
    # )
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

    lambda_step = partial(
        lambda_scheduler,
        lambda_max=cfg.train.lambda_max,
        n_steps=cfg.train.lambda_n_steps,
    )

    if cfg.wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=cfg.model_dump(),
        )

    norm_scaling_factor = estimate_norm_scaling_factor(get_dataloader(cfg, llms, tokenizer), d_model)

    for epoch in range(cfg.train.epochs):
        dataloader = get_dataloader(cfg, llms, tokenizer)  # hacky - this resets the dataloader
        # for step, batch_BMLD in tqdm(enumerate(dataloader)):
        for step, batch_BMLD in enumerate(dataloader):
            optimizer.zero_grad()

            batch_BMLD = batch_BMLD.to(DEVICE)
            batch_BMLD = batch_BMLD * norm_scaling_factor
            _, losses = crosscoder.forward_train(batch_BMLD)

            lambda_ = lambda_step(step=step)
            loss = losses.reconstruction_loss + lambda_ * losses.sparsity_loss

            if (step + 1) % cfg.train.log_every_n_steps == 0:
                log_dict = {
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/loss": loss.item(),
                    "train/reconstruction_loss": losses.reconstruction_loss.item(),
                    "train/sparsity_loss": losses.sparsity_loss.item(),
                    "train/lambda": lambda_,
                }
                print(log_dict)
                if cfg.wandb:
                    wandb.log(log_dict)

            clip_grad_norm_(crosscoder.parameters(), 1.0)
            loss.backward()
            optimizer.step()

        if cfg.train.save_dir and cfg.train.save_every_n_epochs and (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            save_model_and_config(config=cfg, save_dir=cfg.train.save_dir, model=crosscoder, epoch=epoch)


def get_dataloader(
    cfg: Config,
    llms: list[HookedTransformer],
    tokenizer: PreTrainedTokenizerBase,
) -> Iterator[torch.Tensor]:
    activation_harvester = ActivationHarvester(
        hf_dataset=cfg.dataset.hf_dataset,
        cache_dir=cfg.dataset.cache_dir,
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

    return iter(dataloader)


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


def main(config_path: str) -> None:
    config = load_config(Path(config_path))
    train(config)


if __name__ == "__main__":
    fire.Fire(main)
