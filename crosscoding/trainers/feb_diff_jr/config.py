from typing import Any

from crosscoding.log import logger
from crosscoding.trainers.config_common import BaseExperimentConfig
from crosscoding.trainers.jan_update_acausal_crosscoder.config import JanUpdateCrosscoderConfig, TanHSparsityTrainConfig


class JumpReLUModelDiffingFebUpdateCrosscoderConfig(JanUpdateCrosscoderConfig):
    n_shared_latents: int

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.n_shared_latents > self.n_latents:
            raise ValueError(
                "n_shared_latents must be less than or equal to n_latents, "
                f"got {self.n_shared_latents} and {self.n_latents}"
            )


class JumpReLUModelDiffingFebUpdateTrainConfig(TanHSparsityTrainConfig):
    final_lambda_f: float

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if (self.final_lambda_s / self.final_lambda_f) > 0.3:
            logger.warning(
                "final_lambda_s is set to a value that is greater than 30% of final_lambda_f. "
                "Is this intentional? Anthropic use lambda_s / lambda_f ≈ 0.1 - 0.2"
            )


class JumpReLUModelDiffingFebUpdateExperimentConfig(BaseExperimentConfig):
    crosscoder: JumpReLUModelDiffingFebUpdateCrosscoderConfig
    train: JumpReLUModelDiffingFebUpdateTrainConfig
    hookpoint: str
