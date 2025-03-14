from model_diffing.scripts.config_common import BaseExperimentConfig, DataConfig
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateCrosscoderConfig, TanHSparsityTrainConfig


class NoBiasJanUpdateExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: JanUpdateCrosscoderConfig
    train: TanHSparsityTrainConfig
    hookpoints: list[str]
    bias: bool
