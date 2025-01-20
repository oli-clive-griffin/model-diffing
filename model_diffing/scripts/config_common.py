from pydantic import BaseModel


class DatasetConfig(BaseModel):
    hf_dataset: str
    cache_dir: str
    sequence_length: int
    harvest_batch_size: int
    shuffle_buffer_size: int


class WandbConfig(BaseModel):
    name: str | None = None
    project: str
    entity: str


class LLMConfig(BaseModel):
    name: str
    revision: str | None
