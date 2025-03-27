from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig, CrosscoderConfig, DataConfig


class L1CrosscoderConfig(CrosscoderConfig):
    dec_init_norm: float = 0.1


class L1TrainConfig(BaseTrainConfig):
    final_lambda_s: float = 5.0
    lambda_s_n_steps: int = 1000


class L1ExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: L1CrosscoderConfig
    train: L1TrainConfig
    hookpoints: list[str]
