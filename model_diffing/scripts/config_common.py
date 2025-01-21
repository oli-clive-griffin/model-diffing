from typing import Any, Literal

from pydantic import BaseModel


class LLMConfig(BaseModel):
    name: str
    revision: str | None


class LLMsConfig(BaseModel):
    models: list[LLMConfig]
    inference_dtype: str = "float32"


# there's a nicer way to do this with pydantic discriminators but I think it's over the top for now
class SequenceTokensIteratorConfig(BaseModel):
    classname: str
    kwargs: dict[str, Any] | None = None


class ActivationsIteratorConfig(BaseModel):
    layer_indices_to_harvest: list[int]
    harvest_batch_size: int
    sequence_tokens_iterator: SequenceTokensIteratorConfig


class DataConfig(BaseModel):
    activations_iterator: ActivationsIteratorConfig
    shuffle_buffer_size: int
    batch_size: int


class WandbConfig(BaseModel):
    name: str | None = None
    project: str = "model-diffing"
    entity: str = "mars-model-diffing"


class BaseExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    data: DataConfig
    llms: LLMsConfig
    wandb: WandbConfig | Literal["disabled"] = WandbConfig()
