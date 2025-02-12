from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig, DataConfig
from model_diffing.utils import BaseModel


class JumpReLUConfig(BaseModel):
    bandwidth: float = 2.0  # aka ε
    threshold_init: float = 0.1  # aka t
    backprop_through_jumprelu_input: bool = False


class JanUpdateCrosscoderConfig(BaseModel):
    hidden_dim: int
    jumprelu: JumpReLUConfig = JumpReLUConfig()


class JanUpdateTrainConfig(BaseTrainConfig):
    c: float = 4.0

    final_lambda_s: float = 20.0
    """will be linearly ramped from 0 over the entire training run"""

    lambda_p: float = 3e-6


class JanUpdateExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: JanUpdateCrosscoderConfig
    train: JanUpdateTrainConfig
    hookpoints: list[str]
