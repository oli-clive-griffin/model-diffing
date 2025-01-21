import random
from collections.abc import Iterator
from functools import cached_property, partial
from typing import Protocol

import torch
from einops import rearrange
from transformer_lens import HookedTransformer

from model_diffing.utils import chunk


# make me a function?:
class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer],
        layer_indices_to_harvest: list[int],
        batch_size: int,
        sequence_tokens_iterator: Iterator[torch.Tensor],
    ):
        self._llms = llms
        self._layer_indices_to_harvest = layer_indices_to_harvest
        self._batch_size = batch_size
        self._sequence_tokens_iterator = sequence_tokens_iterator

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

    def _get_activations_BSMLD(self, sequence_BS: torch.Tensor) -> torch.Tensor:
        activations = [self._get_model_activations_BSLD(model, sequence_BS) for model in self._llms]
        activations_BSMLD = torch.stack(activations, dim=2)

        return activations_BSMLD

    def get_activations_iterator_BSMLD(self) -> Iterator[torch.Tensor]:
        for sequences_chunk in chunk(self._sequence_tokens_iterator, self._batch_size):
            sequence_tokens_BS = torch.stack(sequences_chunk)
            yield self._get_activations_BSMLD(sequence_tokens_BS)


class ActivationsReshaper(Protocol):
    def __call__(self, activations_iterator_BSMLD: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]: ...


class _ActivationsShuffler:
    def __init__(
        self,
        shuffle_buffer_size: int,
        activations_reshaper: ActivationsReshaper,
        activations_iterator_BSMLD: Iterator[torch.Tensor],
        batch_size: int,
    ):
        self._shuffle_buffer_size = shuffle_buffer_size
        self._activations_reshaper = activations_reshaper
        self._activations_iterator_BSMLD = activations_iterator_BSMLD
        self._batch_size = batch_size

    def get_shuffled_activations_iterator(self) -> Iterator[torch.Tensor]:
        activations_iterator = self._activations_reshaper(self._activations_iterator_BSMLD)

        # this does "waste" the first activation, but this is really not a big deal in the pursuit of simplicity
        activation_shape = next(activations_iterator).shape

        buffer = torch.empty((self._shuffle_buffer_size, *activation_shape))

        available_indices = set()
        stale_indices = set(range(self._shuffle_buffer_size))

        while True:
            # refill buffer
            for stale_idx, activations in zip(list(stale_indices), activations_iterator, strict=False):
                buffer[stale_idx] = activations
                available_indices.add(stale_idx)
                stale_indices.remove(stale_idx)

            # yield batches until buffer is half empty
            while len(available_indices) >= self._shuffle_buffer_size // 2:
                batch_indices = random.sample(list(available_indices), self._batch_size)
                yield buffer[batch_indices]
                available_indices.difference_update(batch_indices)
                stale_indices.update(batch_indices)


# implements `ActivationsReshaper`
def iterate_over_tokens(activations_iterator_BSMLD: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    for activations_BSMLD in activations_iterator_BSMLD:
        activations_BsMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
        yield from activations_BsMLD


# implements `ActivationsReshaper`
def iterate_over_sequences(activations_iterator_BSMLD: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    for activations_BSMLD in activations_iterator_BSMLD:
        yield from activations_BSMLD


TokensActivationsShuffler = partial(_ActivationsShuffler, activations_reshaper=iterate_over_tokens)
