from collections.abc import Iterator
from typing import Any

import pytest
import torch
import wandb
from torch import Tensor

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.models import AcausalCrosscoder, AnthropicTransposeInit, ReLUActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer, validate_num_steps_per_epoch
from model_diffing.scripts.config_common import AdamConfig, BaseTrainConfig
from model_diffing.scripts.train_topk_crosscoder.trainer import aux_loss, topk_dead_latents
from model_diffing.utils import get_device, l2_norm, not_none


class TestTrainer(BaseModelHookpointTrainer[BaseTrainConfig, Any]):
    __test__ = False

    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        return torch.tensor(0.0), None


class FakeActivationsDataloader(BaseModelHookpointActivationsDataloader):
    __test__ = False

    def __init__(
        self,
        batch_size: int = 16,
        n_models: int = 1,
        n_hookpoints: int = 1,
        d_model: int = 16,
        num_batches: int = 100,
    ):
        self._batch_size = batch_size
        self._n_models = n_models
        self._n_hookpoints = n_hookpoints
        self._d_model = d_model
        self._num_batches = num_batches

    def get_activations_iterator_BMPD(self) -> Iterator[Tensor]:
        for _ in range(self._num_batches):
            yield torch.randint(
                0,
                100,
                (self._batch_size, self._n_models, self._n_hookpoints, self._d_model),
                dtype=torch.float32,
            )

    def num_batches(self) -> int | None:
        return self._num_batches

    def get_norm_scaling_factors_MP(self) -> torch.Tensor:
        return torch.ones(self._n_models, self._d_model)


def opt():
    return AdamConfig(learning_rate=1e-3)


@pytest.mark.parametrize(
    "train_cfg",
    [
        BaseTrainConfig(batch_size=1, epochs=2, optimizer=opt()),
        BaseTrainConfig(batch_size=1, epochs=2, num_steps_per_epoch=10, optimizer=opt()),
        BaseTrainConfig(batch_size=1, num_steps=10, optimizer=opt()),
    ],
)
def test_trainer_epochs_steps(train_cfg: BaseTrainConfig) -> None:
    batch_size = 4
    n_models = 1
    hookpoints = ["blocks.0.hook_resid_post"]
    n_hookpoints = len(hookpoints)
    d_model = 16
    num_batches = 10

    activations_dataloader = FakeActivationsDataloader(
        batch_size=batch_size,
        n_models=n_models,
        n_hookpoints=n_hookpoints,
        d_model=d_model,
        num_batches=num_batches,
    )

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        n_latents=16,
        init_strategy=AnthropicTransposeInit(dec_init_norm=0.0),
        activation_fn=ReLUActivation(),
        use_encoder_bias=True,
        use_decoder_bias=True,
    )

    trainer = TestTrainer(
        cfg=train_cfg,
        activations_dataloader=activations_dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb.init(
            name="test",
            project="test",
            mode="disabled",
        ),
        device=get_device(),
        hookpoints=hookpoints,
        save_dir="test_save_dir",
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
    num_steps_per_epoch = validate_num_steps_per_epoch(
        epochs, num_steps_per_epoch, None, activations_dataloader.num_batches()
    )
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
        validate_num_steps_per_epoch(epochs, num_steps_per_epoch, num_steps, activations_dataloader.num_batches())


def test_topk_dead_latents_with_k_less_than_dead():
    preacts_BL = torch.tensor([[5, 4, 3, 2, 1, 0]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 2

    expected_output_BL = torch.tensor([[0, 0, 0, 2, 1, 0]]).float()
    expected_n_used = 2  # k_aux, because it's less than the number of dead features

    aux_latents_BL, n_latents_used = not_none(
        topk_dead_latents(
            pre_activations_BL=preacts_BL,
            dead_features_mask_L=dead_features_mask_L,
            k_aux=k_aux,
        )
    )

    assert torch.allclose(aux_latents_BL, expected_output_BL)
    assert n_latents_used == expected_n_used


def test_topk_dead_latents_with_k_greater_than_dead():
    preacts_BL = torch.tensor([[5, 4, 3, 2, 1, 0]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 4

    expected_output_BL = torch.tensor([[0, 0, 0, 2, 1, 0]]).float()
    expected_n_used = 3  # because there are only 3 dead features (less than k_aux)

    aux_latents_BL, n_latents_used = not_none(
        topk_dead_latents(
            pre_activations_BL=preacts_BL,
            dead_features_mask_L=dead_features_mask_L,
            k_aux=k_aux,
        )
    )

    assert torch.allclose(aux_latents_BL, expected_output_BL)
    assert n_latents_used == expected_n_used


def test_aux_loss_with_k_aux_equal_to_dead():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 3
    decode_BD = lambda x: x  # noqa: E731
    error_BD = torch.tensor([[0, 0, 0, 4, 3, 2]]).float()

    expected_loss = 0.0  # should be able to reconstruct perfectly with the 3 dead features

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=decode_BD,
        error_BXD=error_BD,
    )

    assert loss.item() == expected_loss, f"loss should be {expected_loss}, but is {loss.item()}"


def test_aux_loss_with_k_aux_less_than_dead():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 2
    decode_BD = lambda x: x  # noqa: E731
    # Because of k < n_dead, the "2" should be unused, causing non-zero loss
    error_BD = torch.tensor([[0, 0, 0, 4, 3, 2]]).float()

    # we're using the l2 error mse, so the l2 error is 2 and therefore the loss is 4
    expected_loss = 4.0

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=decode_BD,
        error_BXD=error_BD,
    )

    assert loss.item() == expected_loss, f"loss should be {expected_loss}, but is {loss.item()}"


def test_aux_loss_with_k_aux_more_than_dead():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 1, 1, 1]).bool()
    k_aux = 4
    decode_BD = lambda x: x  # noqa: E731

    error_BD = torch.tensor([[0, 0, 0, 3, 2, 1]]).float()

    expected_aux_topk_BD = torch.tensor([[0, 0, 0, 4, 3, 2]]).float()

    expected_base_loss = l2_norm(error_BD - expected_aux_topk_BD) ** 2
    expected_loss_scale = 3 / 4  # because there are 3 dead features and k_aux is 4
    expected_loss = expected_base_loss * expected_loss_scale

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=decode_BD,
        error_BXD=error_BD,
    )

    assert loss.item() == expected_loss, f"loss should be {expected_loss}, but is {loss.item()}"


def test_aux_loss_with_no_dead_features():
    pre_activations_BL = torch.tensor([[7, 6, 5, 4, 3, 2]]).float()
    dead_features_mask_L = torch.tensor([0, 0, 0, 0, 0, 0]).bool()
    k_aux = 4

    loss = aux_loss(
        pre_activations_BL=pre_activations_BL,
        dead_features_mask_L=dead_features_mask_L,
        k_aux=k_aux,
        decode_BXD=None,  # type: ignore
        error_BXD=None,  # type: ignore
    )

    assert loss.item() == 0.0
