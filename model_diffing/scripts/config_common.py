from operator import xor
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from model_diffing.data.activation_harvester import CacheMode
from model_diffing.utils import BaseModel


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
    betas: tuple[float, float] = (0.9, 0.999)


class ScheduleFreeSigNumConfig(BaseModel):
    type: Literal["schedule_free_signum"] = "schedule_free_signum"
    learning_rate: float = 1e-3
    momentum: float = 0.95


OptimizerCfg = AdamConfig | ScheduleFreeSigNumConfig
# Specific config classes for each loader type
class HuggingfaceTextDatasetConfig(BaseModel):
    type: Literal["HuggingfaceTextDatasetTokenSequenceLoader"] = "HuggingfaceTextDatasetTokenSequenceLoader"
    hf_dataset_name: str
    sequence_length: int
    shuffle_buffer_size: int | None = None


class ConnorGemma2Config(BaseModel):
    type: Literal["ConnorGemma2TokenSequenceLoader"] = "ConnorGemma2TokenSequenceLoader"
    # No additional parameters needed


class ToyOverfittingConfig(BaseModel):
    type: Literal["ToyOverfittingTokenSequenceLoader"] = "ToyOverfittingTokenSequenceLoader"
    sequence_length: int
    vocab_size: int = 10
    first_n_tokens_special: int = 2


class MathDatasetConfig(BaseModel):
    type: Literal["MathDatasetTokenSequenceLoader"] = "MathDatasetTokenSequenceLoader"
    max_sequence_length: int
    include_base_answers: bool = False
    include_reasoning_answers: bool = False


class ActivationsHarvesterConfig(BaseModel):
    llms: list[LLMConfig]
    inference_dtype: str = "float32"
    harvesting_batch_size: int
    cache_mode: CacheMode = "no_cache"


SequenceIteratorCfg = HuggingfaceTextDatasetConfig | ConnorGemma2Config | ToyOverfittingConfig | MathDatasetConfig


class DataConfig(BaseModel):
    sequence_iterator: SequenceIteratorCfg = Field(discriminator="type")
    activations_harvester: ActivationsHarvesterConfig
    activations_shuffle_buffer_size: int | None = None
    """if this is None, we will not shuffle the activations"""
    n_tokens_for_norm_estimate: int


class BaseTrainConfig(BaseModel):
    batch_size: int
    optimizer: OptimizerCfg = Field(discriminator="type", default_factory=AdamConfig)
    epochs: int | None = None
    num_steps_per_epoch: int | None = None
    num_steps: int | None = None
    save_every_n_steps: int | None = None
    log_every_n_steps: int | None = None
    gradient_accumulation_steps_per_batch: int = 1  

    def minibatch_size(self) -> int:
        return self.batch_size // self.gradient_accumulation_steps_per_batch

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not xor(
            (self.epochs is None and self.num_steps_per_epoch is None),
            (self.num_steps is None),
        ):
            raise ValueError("must provide either only epochs and num_steps_per_epoch or only num_steps")

        if self.batch_size % self.gradient_accumulation_steps_per_batch != 0:
            raise ValueError("batch_size must be divisible by gradient_accumulation_steps_per_batch")


class WandbConfig(BaseModel):
    entity: str = "mars-model-diffing"
    project: str = "model-diffing"
    mode: Literal["disabled", "online", "offline"] = "online"

class BaseExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    base_save_dir: str = ".checkpoints"
    wandb: WandbConfig= WandbConfig()
    experiment_name: str

    @property
    def save_dir(self) -> Path:
        assert self.experiment_name is not None
        return Path(self.base_save_dir) / self.experiment_name
