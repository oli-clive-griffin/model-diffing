from crosscoding.trainers.config_common import BaseExperimentConfig, BaseTrainConfig, CrosscoderConfig


class TopkSkipTransCrosscoderConfig(CrosscoderConfig):
    dec_init_norm: float = 0.1
    k: int


class TopkSkipTransCrosscoderExperimentConfig(BaseExperimentConfig):
    crosscoder: TopkSkipTransCrosscoderConfig
    train: BaseTrainConfig
    mlp_indices: list[int]
