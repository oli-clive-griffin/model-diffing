from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig, DataConfig
from model_diffing.utils import BaseModel


class TopkSkipTransCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int


class TopkSkipTransCrosscoderExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: TopkSkipTransCrosscoderConfig
    train: BaseTrainConfig
    mlp_indices: list[int]
