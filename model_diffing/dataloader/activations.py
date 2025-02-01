from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property

import torch
from einops import rearrange
from transformer_lens import HookedTransformer  # type: ignore

from model_diffing.dataloader.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.dataloader.token_loader import TokenSequenceLoader
from model_diffing.log import logger
from model_diffing.scripts.utils import estimate_norm_scaling_factor_ML


class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer],
        layer_indices_to_harvest: list[int],
    ):
        self._llms = llms
        self._layer_indices_to_harvest = layer_indices_to_harvest

    @cached_property
    def names(self) -> list[str]:
        return [f"blocks.{num}.hook_resid_post" for num in self._layer_indices_to_harvest]

    @cached_property
    def names_set(self) -> set[str]:
        return set(self.names)

    def _names_filter(self, name: str) -> bool:
        return name in self.names_set

    def _get_model_activations_BSLD(self, model: HookedTransformer, sequence_BS: torch.Tensor) -> torch.Tensor:
        _, cache = model.run_with_cache(sequence_BS, names_filter=self._names_filter)
        activations_BSLD = torch.stack([cache[name] for name in self.names], dim=2)  # adds layer dim (L)
        # cropped_activations_BSLD = activations_BSLD[:, 1:, :, :]  # remove BOS, need
        return activations_BSLD

    def get_activations_BSMLD(self, sequence_BS: torch.Tensor) -> torch.Tensor:
        activations = [self._get_model_activations_BSLD(model, sequence_BS) for model in self._llms]
        activations_BSMLD = torch.stack(activations, dim=2)
        return activations_BSMLD

    def activation_shape_MLD(self) -> tuple[int, int, int]:
        return (
            len(self._layer_indices_to_harvest),
            len(self._llms),
            self._llms[0].cfg.d_model,
        )


class BaseActivationsDataloader(ABC):
    @abstractmethod
    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def batch_shape_BMLD(self) -> tuple[int, int, int, int]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...


class ScaledActivationsDataloader(BaseActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int,
        yield_batch_size: int,
        device: torch.device,
        n_batches_for_norm_estimate: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size

        self.norm_scaling_factors_ML = estimate_norm_scaling_factor_ML(
            self._get_shuffled_raw_activations_iterator_BMLD(),
            device,
            n_batches_for_norm_estimate,
        )

    @torch.no_grad()
    def _activations_iterator_MLD(self) -> Iterator[torch.Tensor]:
        for sequences_chunk_BS in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMLD = self._activations_harvester.get_activations_BSMLD(sequences_chunk_BS)
            activations_BsMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
            yield from activations_BsMLD

    @torch.no_grad()
    def _get_shuffled_raw_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        activations_iterator_MLD = self._activations_iterator_MLD()

        # shuffle these token activations, so that we eliminate high feature correlations inside sequences
        shuffled_activations_iterator_BMLD = batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=activations_iterator_MLD,
            shuffle_buffer_size=self._activations_shuffle_buffer_size,
            yield_batch_size=self._yield_batch_size,
        )

        return shuffled_activations_iterator_BMLD

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def batch_shape_BMLD(self) -> tuple[int, int, int, int]:
        return (
            self._yield_batch_size,
            *self._activations_harvester.activation_shape_MLD(),
        )

    @torch.no_grad()
    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        raw_activations_iterator_BMLD = self._get_shuffled_raw_activations_iterator_BMLD()
        scaling_factors_ML1 = rearrange(self.norm_scaling_factors_ML, "m l -> m l 1")
        for unscaled_example_BMLD in raw_activations_iterator_BMLD:
            yield unscaled_example_BMLD * scaling_factors_ML1
