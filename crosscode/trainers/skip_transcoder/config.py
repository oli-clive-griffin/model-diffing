from crosscode.trainers.config_common import BaseExperimentConfig, BaseSparseCoder
from crosscode.trainers.train_topk_crosscoder.config import TopKTrainConfig


class TopkSkipTranscoderConfig(BaseSparseCoder):
    dec_init_norm: float = 0.1
    k: int
    linear_skip: bool = True


class TopkSkipTranscoderExperimentConfig(BaseExperimentConfig):
    transcoder: TopkSkipTranscoderConfig
    train: TopKTrainConfig
    mlp_index_in: int
    mlp_indices_out: list[int]
