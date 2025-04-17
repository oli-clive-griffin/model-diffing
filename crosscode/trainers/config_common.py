import os
from operator import xor
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from crosscode.data.activation_harvester import CacheMode
from crosscode.utils import BaseModel


class HuggingfaceTextDatasetConfig(BaseModel):
    hf_dataset_name: str = "monology/pile-uncopyrighted"
    sequence_length: int = 2048
    sequences_shuffle_buffer_size: int | None = None


class LLMConfig(BaseModel):
    name: str | None = None
    revision: str | None = None

    base_archicteture_name: str | None = None
    hf_model_name: str | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        assert xor(
            (self.name is not None),
            (self.base_archicteture_name is not None and self.hf_model_name is not None),
        ), (
            "must provide either name (to load from official model list)"
            " or base_archicteture_name and hf_model_name (to load from huggingface)"
        )


class AdamConfig(BaseModel):
    type: Literal["adam"] = "adam"
    warmup_pct: float = 0.05
    learning_rate: float = 5e-5
    warmdown_pct: float = 0.2
    betas: tuple[float, float] = (0.0, 0.999)  # beta0 = 0.0 seems to work better in many peoples' experience


class ScheduleFreeSigNumConfig(BaseModel):
    type: Literal["schedule_free_signum"] = "schedule_free_signum"
    learning_rate: float = 1e-3
    momentum: float = 0.95


OptimizerCfg = AdamConfig | ScheduleFreeSigNumConfig


class ActivationsHarvesterConfig(BaseModel):
    llms: list[LLMConfig]
    inference_dtype: str = "float32"
    harvesting_batch_size: int
    cache_mode: CacheMode = "no_cache"


class DataConfig(BaseModel):
    token_sequence_loader: HuggingfaceTextDatasetConfig = HuggingfaceTextDatasetConfig()
    activations_harvester: ActivationsHarvesterConfig
    n_tokens_for_norm_estimate: int = 100_000
    activations_shuffle_buffer_size: int | None = None


class BaseTrainConfig(BaseModel):
    batch_size: int
    optimizer: OptimizerCfg = Field(discriminator="type", default_factory=AdamConfig)
    num_steps: int
    save_every_n_steps: int | None = None
    upload_saves_to_wandb: bool = False
    log_every_n_steps: int
    gradient_accumulation_microbatches_per_step: int = 1

    def minibatch_size(self) -> int:
        return self.batch_size // self.gradient_accumulation_microbatches_per_step

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.batch_size % self.gradient_accumulation_microbatches_per_step != 0:
            raise ValueError("batch_size must be divisible by gradient_accumulation_steps_per_batch")


class BaseSparseCoder(BaseModel):
    n_latents: int
    use_encoder_bias: bool = True
    use_decoder_bias: bool = True


class WandbConfig(BaseModel):
    entity: str | None = None
    project: str | None = None
    mode: Literal["disabled", "online", "offline"] = "online"

    @field_validator("entity")
    @classmethod
    def validate_entity(cls, v: str | None) -> str:
        if v is None:
            v = os.environ.get("WANDB_ENTITY")
        if v is None:
            raise ValueError("WANDB_ENTITY must be set if 'entity' is not provided")
        return v

    @field_validator("project")
    @classmethod
    def validate_project(cls, v: str | None) -> str:
        if v is None:
            v = os.environ.get("WANDB_PROJECT")
        if v is None:
            raise ValueError("WANDB_PROJECT must be set if 'project' is not provided")
        return v


class BaseExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    base_save_dir: str = ".checkpoints"
    wandb: WandbConfig = WandbConfig()
    experiment_name: str
    data: DataConfig

    @property
    def save_dir(self) -> Path:
        assert self.experiment_name is not None
        return Path(self.base_save_dir) / self.experiment_name
