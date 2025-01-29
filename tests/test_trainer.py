from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.models.crosscoder import build_relu_crosscoder
from model_diffing.scripts.config_common import AdamDecayTo0LearningRateConfig, BaseTrainConfig
from model_diffing.scripts.trainer import BaseTrainer, validate_num_steps_per_epoch
from model_diffing.utils import get_device


class TestTrainer(BaseTrainer[BaseTrainConfig]):
    __test__ = False

    def _train_step(self, batch_BMLD: Tensor) -> dict[str, float]:
        return {
            "loss": 0.0,
        }


class FakeActivationsDataloader(BaseActivationsDataloader):
    __test__ = False

    def __init__(
        self,
        batch_size: int = 16,
        n_models: int = 1,
        n_layers: int = 1,
        d_model: int = 16,
        num_batches: int = 100,
    ):
        self._batch_size = batch_size
        self._n_models = n_models
        self._n_layers = n_layers
        self._d_model = d_model
        self._num_batches = num_batches

    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        for _ in range(self._num_batches):
            yield torch.randint(
                0,
                100,
                (self._batch_size, self._n_layers, self._n_models, self._d_model),
                dtype=torch.float32,
            )

    def batch_shape_BMLD(self) -> tuple[int, int, int, int]:
        return (self._batch_size, self._n_layers, self._n_models, self._d_model)

    def num_batches(self) -> int | None:
        return self._num_batches


def opt():
    return AdamDecayTo0LearningRateConfig(initial_learning_rate=1e-3)


@pytest.mark.parametrize(
    "train_cfg",
    [
        BaseTrainConfig(epochs=2, optimizer=opt()),
        BaseTrainConfig(epochs=2, num_steps_per_epoch=10, optimizer=opt()),
        BaseTrainConfig(num_steps=10, optimizer=opt()),
    ],
)
def test_trainer_epochs_steps(train_cfg: BaseTrainConfig) -> None:
    batch_size = 4
    n_models = 1
    layer_indices_to_harvest = [0]
    n_layers = len(layer_indices_to_harvest)
    d_model = 16
    num_batches = 10

    activations_dataloader = FakeActivationsDataloader(
        batch_size=batch_size,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        num_batches=num_batches,
    )

    crosscoder = build_relu_crosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=16,
        dec_init_norm=0.0,
    )

    trainer = TestTrainer(
        cfg=train_cfg,
        activations_dataloader=activations_dataloader,
        crosscoder=crosscoder,
        wandb_run=None,
        device=get_device(),
        layers_to_harvest=layer_indices_to_harvest,
        experiment_name="test",
    )

    trainer.train()


@pytest.mark.parametrize(
    "epochs, num_steps_per_epoch, dataloader_num_batches, expected",
    [
        # WHEN num_steps_per_epoch < dataloader_num_batches (should return num_steps_per_epoch)
        (10, 100, 200, 100),
        # WHEN num_steps_per_epoch > dataloader_num_batches (should return dataloader_num_batches)
        (10, 200, 100, 100),
        # WHEN epochs is provided but num_steps_per_epoch is not (should return dataloader_num_batches)
        (10, None, 100, 100),
    ],
)
def test_validate_num_steps_per_epoch_happy_path(
    epochs: int,
    num_steps_per_epoch: int | None,
    dataloader_num_batches: int,
    expected: int,
) -> None:
    activations_dataloader = FakeActivationsDataloader(num_batches=dataloader_num_batches)
    num_steps_per_epoch = validate_num_steps_per_epoch(epochs, num_steps_per_epoch, None, activations_dataloader)
    assert num_steps_per_epoch == expected


@pytest.mark.parametrize(
    "epochs, num_steps_per_epoch, num_steps, should_raise",
    [
        # WHEN num_steps_per_epoch is provided but not epochs
        (None, 100, None, ValueError),
        # WHEN both epochs and num_steps are provided
        (10, None, 100, ValueError),
        # WHEN neither epochs nor num_steps are provided
        (None, None, None, ValueError),
    ],
)
def test_validate_num_steps_per_epoch_errors(
    epochs: int | None,
    num_steps_per_epoch: int | None,
    num_steps: int | None,
    should_raise: type[Exception],
) -> None:
    activations_dataloader = FakeActivationsDataloader(num_batches=99999)

    with pytest.raises(should_raise):
        validate_num_steps_per_epoch(epochs, num_steps_per_epoch, num_steps, activations_dataloader)
