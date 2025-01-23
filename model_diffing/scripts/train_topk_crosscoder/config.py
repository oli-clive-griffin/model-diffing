from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from model_diffing.scripts.config_common import (
    AdamDecayTo0LearningRateConfig,
    DataConfig,
    LLMsConfig,
    WandbConfig,
)


class TrainConfig(BaseModel):
    optimizer: AdamDecayTo0LearningRateConfig
    num_steps: int
    save_dir: Path | None
    save_every_n_steps: int | None
    log_every_n_steps: int
    n_batches_for_norm_estimate: int = 100


class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int


class TopKExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    data: DataConfig
    llms: LLMsConfig
    wandb: WandbConfig | Literal["disabled"] = WandbConfig()
    crosscoder: TopKCrosscoderConfig
    train: TrainConfig
