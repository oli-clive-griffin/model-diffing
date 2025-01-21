from pathlib import Path

from pydantic import BaseModel

from model_diffing.scripts.config_common import BaseExperimentConfig


class DecayTo0LearningRateConfig(BaseModel):
    initial_learning_rate: float
    last_pct_of_steps: float = 0.2


class TrainConfig(BaseModel):
    learning_rate: DecayTo0LearningRateConfig
    lambda_max: float = 5.0
    lambda_n_steps: int = 1000
    num_steps: int
    save_dir: Path | None
    save_every_n_steps: int | None
    log_every_n_steps: int
    n_batches_for_norm_estimate: int = 100


class L1CrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1


class L1ExperimentConfig(BaseExperimentConfig):
    crosscoder: L1CrosscoderConfig
    train: TrainConfig
