from typing import Any

from pydantic import BaseModel


class LLMConfig(BaseModel):
    name: str
    revision: str | None = None


class AdamDecayTo0LearningRateConfig(BaseModel):
    initial_learning_rate: float
    last_pct_of_steps: float = 0.2


# there's a nicer way to do this with pydantic discriminators but I think it's over the top for now
class SequenceIteratorConfig(BaseModel):
    batch_size: int
    classname: str
    kwargs: dict[str, Any] | None = None


class ActivationsHarvesterConfig(BaseModel):
    llms: list[LLMConfig]
    layer_indices_to_harvest: list[int]
    inference_dtype: str = "float32"


class DataConfig(BaseModel):
    sequence_iterator: SequenceIteratorConfig
    activations_harvester: ActivationsHarvesterConfig
    activations_shuffle_buffer_size: int
    cc_training_batch_size: int


class BaseTrainConfig(BaseModel):
    optimizer: AdamDecayTo0LearningRateConfig
    epochs: int | None = None
    num_steps_per_epoch: int | None = None
    num_steps: int | None = None
    base_save_dir: str | None = ".checkpoints"
    save_every_n_steps: int | None = None
    log_every_n_steps: int | None = None
    log_visualizations_every_n_steps: int | None = None
    n_batches_for_norm_estimate: int = 100

    def __post_init__(self):
        if not (
            (
                self.epochs is not None  #
                and self.num_steps_per_epoch is not None
                and self.num_steps is None
            )
            or (
                self.epochs is None  #
                and self.num_steps_per_epoch is None
                and self.num_steps is not None
            )
        ):
            raise ValueError("must provide either only epochs and num_steps_per_epoch or only num_steps")


class BaseExperimentConfig(BaseModel):
    seed: int = 42
    cache_dir: str = ".cache"
    data: DataConfig
    wandb: bool = False
    experiment_name: str


__DEMO = BaseExperimentConfig(
    data=DataConfig(
        sequence_iterator=SequenceIteratorConfig(
            batch_size=16,
            classname="CommonCorpusTokenSequenceIterator",
            kwargs={
                "sequence_length": 258,
                "shuffle_buffer_size": 16_384,
            },
        ),
        activations_harvester=ActivationsHarvesterConfig(
            llms=[
                LLMConfig(
                    name="gpt2",
                ),
            ],
            layer_indices_to_harvest=[0, 3, 7, 9, 11],
        ),
        activations_shuffle_buffer_size=1000,
        cc_training_batch_size=16,
    ),
    wandb=False,
    experiment_name="demo",
)
