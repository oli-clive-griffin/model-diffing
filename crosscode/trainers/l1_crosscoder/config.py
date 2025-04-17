from crosscode.trainers.config_common import BaseExperimentConfig, BaseSparseCoder, BaseTrainConfig


class L1CrosscoderConfig(BaseSparseCoder):
    dec_init_norm: float = 0.1


class L1TrainConfig(BaseTrainConfig):
    final_lambda_s: float = 5.0
    lambda_s_num_steps: int = 1000


class L1ExperimentConfig(BaseExperimentConfig):
    crosscoder: L1CrosscoderConfig
    train: L1TrainConfig
    hookpoints: list[str]
