from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig, DataConfig
from model_diffing.utils import BaseModel


class L1CrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1


class L1TrainConfig(BaseTrainConfig):
    lambda_s_max: float = 5.0
    lambda_s_n_steps: int = 1000


class L1ExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: L1CrosscoderConfig
    train: L1TrainConfig
    hookpoints: list[str]
