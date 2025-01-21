from pathlib import Path

from pydantic import BaseModel

from model_diffing.scripts.config_common import BaseExperimentConfig


class DecayTo0LearningRateConfig(BaseModel):
    initial_learning_rate: float
    last_pct_of_steps: float = 0.2


class TrainConfig(BaseModel):
    learning_rate: DecayTo0LearningRateConfig
    num_steps: int
    save_dir: Path | None
    save_every_n_steps: int | None
    log_every_n_steps: int
    n_batches_for_norm_estimate: int = 100


class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int


class TopKExperimentConfig(BaseExperimentConfig):
    crosscoder: TopKCrosscoderConfig
    train: TrainConfig
