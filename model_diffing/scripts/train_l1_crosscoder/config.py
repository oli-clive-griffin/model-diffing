from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from model_diffing.scripts.config_common import AdamDecayTo0LearningRateConfig, DataConfig, LLMsConfig, WandbConfig


class TrainConfig(BaseModel):
    optimizer: AdamDecayTo0LearningRateConfig
    l1_coef_max: float = 5.0
    l1_coef_n_steps: int = 1000
    num_steps: int
    save_dir: Path | None
    save_every_n_steps: int | None
    log_every_n_steps: int
    n_batches_for_norm_estimate: int = 100


class L1CrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1


class L1ExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    data: DataConfig
    llms: LLMsConfig
    wandb: WandbConfig | Literal["disabled"] = WandbConfig()
    crosscoder: L1CrosscoderConfig
    train: TrainConfig
