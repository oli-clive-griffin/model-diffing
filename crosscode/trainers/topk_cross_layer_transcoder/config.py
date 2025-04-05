from crosscode.trainers.config_common import BaseExperimentConfig, BaseSparseCoder
from crosscode.trainers.topk_crosscoder.config import TopKTrainConfig


class TopkCrossLayerTranscoderConfig(BaseSparseCoder):
    dec_init_norm: float = 0.1
    k: int
    linear_skip: bool = False


class TopkCrossLayerTranscoderExperimentConfig(BaseExperimentConfig):
    transcoder: TopkCrossLayerTranscoderConfig
    train: TopKTrainConfig
    in_hookpoint: str
    out_hookpoints: list[str]
