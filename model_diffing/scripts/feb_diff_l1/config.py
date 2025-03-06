from typing import Any

from model_diffing.log import logger
from model_diffing.scripts.config_common import BaseExperimentConfig, DataConfig
from model_diffing.scripts.train_l1_crosscoder.config import L1CrosscoderConfig, L1TrainConfig


class L1ModelDiffingFebUpdateCrosscoderConfig(L1CrosscoderConfig):
    n_shared_latents: int

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.n_shared_latents > self.hidden_dim:
            raise ValueError(
                "n_shared_latents must be less than or equal to hidden_dim, "
                f"got {self.n_shared_latents} and {self.hidden_dim}"
            )


class L1ModelDiffingFebUpdateTrainConfig(L1TrainConfig):
    final_lambda_f: float

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.final_lambda_f / self.final_lambda_s > 0.3:
            logger.warning(
                "final_lambda_f is set to a value that is greater than 30% of final_lambda_s. "
                "Is this intentional? Anthropic use lambda_s / lambda_f ≈ 0.1 - 0.2"
            )


class L1ModelDiffingFebUpdateExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: L1ModelDiffingFebUpdateCrosscoderConfig
    train: L1ModelDiffingFebUpdateTrainConfig
    hookpoints: list[str]
