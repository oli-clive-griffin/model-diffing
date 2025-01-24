from pydantic import BaseModel

from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig


class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int


class TopKExperimentConfig(BaseExperimentConfig):
    crosscoder: TopKCrosscoderConfig
    train: BaseTrainConfig
