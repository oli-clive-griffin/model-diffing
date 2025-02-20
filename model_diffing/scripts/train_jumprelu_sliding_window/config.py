from model_diffing.scripts.config_common import BaseExperimentConfig, DataConfig
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateCrosscoderConfig, JanUpdateTrainConfig


class SlidingWindowExperimentConfig(BaseExperimentConfig):
    crosscoder: JanUpdateCrosscoderConfig
    train: JanUpdateTrainConfig
    data: DataConfig
    hookpoints: list[str]
