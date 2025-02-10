from datetime import datetime
from operator import xor
from typing import Any

from model_diffing.utils import BaseModel


class LLMConfig(BaseModel):
    name: str
    revision: str | None = None


class AdamDecayTo0LearningRateConfig(BaseModel):
    warmup_pct: float = 0.05
    initial_learning_rate: float
    last_pct_of_steps: float = 0.2


# there's a nicer way to do this with pydantic discriminators but I think it's over the top for now
class SequenceIteratorConfig(BaseModel):
    classname: str
    kwargs: dict[str, Any] | None = None


class ActivationsHarvesterConfig(BaseModel):
    llms: list[LLMConfig]
    layer_indices_to_harvest: list[int]
    inference_dtype: str = "float32"
    harvesting_batch_size: int


class DataConfig(BaseModel):
    sequence_iterator: SequenceIteratorConfig
    activations_harvester: ActivationsHarvesterConfig
    activations_shuffle_buffer_size: int
    n_batches_for_norm_estimate: int = 100


class BaseTrainConfig(BaseModel):
    batch_size: int
    optimizer: AdamDecayTo0LearningRateConfig
    epochs: int | None = None
    num_steps_per_epoch: int | None = None
    num_steps: int | None = None
    base_save_dir: str = ".checkpoints"
    save_every_n_steps: int | None = None
    log_every_n_steps: int | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not xor(
            (self.epochs is None and self.num_steps_per_epoch is None),
            (self.num_steps is None),
        ):
            raise ValueError("must provide either only epochs and num_steps_per_epoch or only num_steps")


class BaseExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    wandb: bool
    experiment_name: str

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.experiment_name = f"{self.experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        print(f"experiment_name: {self.experiment_name}")
