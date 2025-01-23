from typing import Any

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
