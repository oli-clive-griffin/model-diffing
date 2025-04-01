from crosscoding.models.activations.topk import TopKStyle
from crosscoding.trainers.config_common import BaseExperimentConfig, BaseTrainConfig, CrosscoderConfig


class TopKCrosscoderConfig(CrosscoderConfig):
    dec_init_norm: float = 0.1
    k: int


class TopKTrainConfig(BaseTrainConfig):
    topk_style: TopKStyle
    dead_latents_threshold_n_examples: int = 1_000_000
    lambda_aux: float = 1 / 32
    k_aux: int | None = None
    """see heuristic in appendix B.1 in 'scaling and evaluating sparse autoencoders'"""


class TopKExperimentConfig(BaseExperimentConfig):
    crosscoder: TopKCrosscoderConfig
    train: TopKTrainConfig
    hookpoints: list[str]
