from model_diffing.scripts.config_common import BaseExperimentConfig
from model_diffing.scripts.train_l1_crosscoder.config import L1CrosscoderConfig, L1TrainConfig


class L1SlidingWindowExperimentConfig(BaseExperimentConfig):
    crosscoder: L1CrosscoderConfig
    train: L1TrainConfig
    hookpoints: list[str]
