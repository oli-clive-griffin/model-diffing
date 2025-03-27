from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig, CrosscoderConfig, DataConfig


class TopkSkipTransCrosscoderConfig(CrosscoderConfig):
    dec_init_norm: float = 0.1
    k: int


class TopkSkipTransCrosscoderExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: TopkSkipTransCrosscoderConfig
    train: BaseTrainConfig
    mlp_indices: list[int]
