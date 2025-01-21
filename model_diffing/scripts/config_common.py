from typing import Literal

from pydantic import BaseModel


class LLMConfig(BaseModel):
    name: str
    revision: str | None


class LLMsConfig(BaseModel):
    models: list[LLMConfig]
    inference_dtype: str = "float32"


class SequenceTokensIteratorConfig(BaseModel):
    name: Literal["common_corpus", "connor_gemma"]
    sequence_length: int | None


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
