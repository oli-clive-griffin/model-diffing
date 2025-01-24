from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class LLMConfig(BaseModel):
    name: str
    revision: str | None = None


class LLMsConfig(BaseModel):
    models: list[LLMConfig]
    inference_dtype: str = "float32"


class AdamDecayTo0LearningRateConfig(BaseModel):
    initial_learning_rate: float
    last_pct_of_steps: float = 0.2


# there's a nicer way to do this with pydantic discriminators but I think it's over the top for now
class SequenceIteratorConfig(BaseModel):
    classname: str
    kwargs: dict[str, Any] | None = None


class ActivationsHarvesterConfig(BaseModel):
    llms: LLMsConfig
    layer_indices_to_harvest: list[int]
    harvest_batch_size: int


class DataConfig(BaseModel):
    sequence_iterator: SequenceIteratorConfig
    sequence_shuffle_buffer_size: int
    activations_harvester: ActivationsHarvesterConfig
    activations_shuffle_buffer_size: int
    cc_training_batch_size: int


class WandbConfig(BaseModel):
    name: str | None = None
    project: str = "model-diffing"
    entity: str = "mars-model-diffing"


class BaseTrainConfig(BaseModel):
    optimizer: AdamDecayTo0LearningRateConfig
    num_steps: int
    save_dir: Path | None
    save_every_n_steps: int | None
    log_every_n_steps: int
    log_visualizations_every_n_steps: int
    n_batches_for_norm_estimate: int = 100


class BaseExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    data: DataConfig
    wandb: WandbConfig | Literal["disabled"] = WandbConfig()
