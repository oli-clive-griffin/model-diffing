import random
from collections.abc import Iterator
from functools import cached_property
from typing import Any, cast

import torch
from datasets import load_dataset
from einops import rearrange
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.utils import chunk


class ActivationHarvester:
    def __init__(
        self,
        hf_dataset: str,
        cache_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        models: list[HookedTransformer],
        layer_indices_to_harvest: list[int],
        batch_size: int,
        sequence_length: int,
    ):
        self._hf_dataset = hf_dataset
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._models = models
        self._layer_indices_to_harvest = layer_indices_to_harvest
        self._batch_size = batch_size
        self._sequence_length = sequence_length

        assert len({model.cfg.d_model for model in self._models}) == 1, "All models must have the same d_model"
        self._d_model = self._models[0].cfg.d_model

    @cached_property
    def names(self) -> list[str]:
        return [f"blocks.{num}.hook_resid_post" for num in self._layer_indices_to_harvest]

    @cached_property
    def names_set(self) -> set[str]:
        return set(self.names)

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def n_models(self) -> int:
        return len(self._models)

    @property
    def n_layers(self) -> int:
        return len(self._layer_indices_to_harvest)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def _activation_shape_BSMLD(self) -> tuple[int, int, int, int, int]:
        return (
            self._batch_size,
            self._sequence_length,
            self.n_models,
            self.n_layers,
            self.d_model,
        )

    def _names_filter(self, name: str) -> bool:
        return name in self.names_set

    def _sequence_iterator(self) -> Iterator[torch.Tensor]:
        dataset = load_dataset(self._hf_dataset, streaming=True, cache_dir=self._cache_dir)

        for example in cast(Any, dataset)["train"]:
            seq_tokens_S = torch.tensor(self._tokenizer(example["text"])["input_ids"])
            assert len(seq_tokens_S.shape) == 1, f"seq_tokens_S.shape should be 1D but was {seq_tokens_S.shape}"
            num_full_sequences = len(seq_tokens_S) // self._sequence_length
            if num_full_sequences == 0:
                continue

            for i in range(0, num_full_sequences * self._sequence_length, self._sequence_length):
                yield seq_tokens_S[i : i + self._sequence_length]

    def _get_model_activations_BSLD(self, model: HookedTransformer, sequence_BS: torch.Tensor) -> torch.Tensor:
        _, cache = model.run_with_cache(sequence_BS, names_filter=self._names_filter)
        activations_BSLD = torch.stack([cache[name] for name in self.names], dim=2)  # add layer dim (L)
        # cropped_activations_BSLD = activations_BSLD[:, 1:, :, :]  # remove BOS, need
        return activations_BSLD

    def get_activations_iterator_BSMLD(self) -> Iterator[torch.Tensor]:
        for sequences_chunk in chunk(self._sequence_iterator(), self._batch_size):
            sequence_BS = torch.stack(sequences_chunk)

            assert sequence_BS.shape == (self._batch_size, self._sequence_length), (
                f"sequence_BS.shape should be {(self._batch_size, self._sequence_length)} but was {sequence_BS.shape}"
            )

            activations = [self._get_model_activations_BSLD(model, sequence_BS) for model in self._models]
            activations_BSMLD = torch.stack(activations, dim=2)

            assert activations_BSMLD.shape == self._activation_shape_BSMLD, (
                f"activations_BSMLD.shape should be {self._activation_shape_BSMLD} but was {activations_BSMLD.shape}"
            )

            yield activations_BSMLD


class ShuffledTokensActivationsLoader:
    """
    takes activations from an ActivationHarvester, flattens across batch and
    sequence dimensions (because this is not important for crosscoder training)
    and shuffles them using a shuffle buffer.
    """

    def __init__(
        self,
        activation_harvester: ActivationHarvester,
        shuffle_buffer_size: int,
        batch_size: int,
    ):
        self._activation_harvester = activation_harvester
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

        self._activation_shape_MLD = (
            self._activation_harvester.n_models,
            self._activation_harvester.n_layers,
            self._activation_harvester.d_model,
        )

    def _get_activations_iterator_MLD(self) -> Iterator[torch.Tensor]:
        for activations_BSMLD in self._activation_harvester.get_activations_iterator_BSMLD():
            activations_BsMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
            yield from activations_BsMLD
            # If this "yield from" is hard to understand, it is equivalent to:
            # ```
            # for activations_MLD in activations_BsMLD:
            #     yield activations_MLD
            # ```

    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        if self._batch_size > self._shuffle_buffer_size // 2:
            raise ValueError(
                f"Batch size cannot be greater than half the buffer size, {self._batch_size} > {self._shuffle_buffer_size // 2}"
            )

        buffer_BMLD = torch.empty((self._shuffle_buffer_size, *self._activation_shape_MLD))

        available_indices = set()
        stale_indices = set(range(self._shuffle_buffer_size))

        iterator_MLD = self._get_activations_iterator_MLD()

        while True:
            # refill buffer
            for stale_idx, activation_MLD in zip(list(stale_indices), iterator_MLD, strict=False):
                assert activation_MLD.shape == self._activation_shape_MLD, (
                    f"activation_MLD.shape should be {self._activation_shape_MLD} but was {activation_MLD.shape}"
                )
                buffer_BMLD[stale_idx] = activation_MLD
                available_indices.add(stale_idx)
                stale_indices.remove(stale_idx)

            # yield batches until buffer is half empty
            while len(available_indices) >= self._shuffle_buffer_size // 2:
                batch_indices = random.sample(list(available_indices), self._batch_size)
                yield buffer_BMLD[batch_indices]
                available_indices.difference_update(batch_indices)
                stale_indices.update(batch_indices)


class ShuffledSequenceActivationsLoader:
    """
    takes activations from an ActivationHarvester, and shuffles sequences of activations using a
    shuffle buffer. Importantly, this doesn't shuffle across sequence dimensions, it leaves sequences
    in the same order as they were in the original dataset. Just shuffles across the batch dimension.
    """

    def __init__(
        self,
        activation_harvester: ActivationHarvester,
        shuffle_buffer_size: int,
        batch_size: int,
    ):
        self._activation_harvester = activation_harvester
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

        self._activation_shape_SMLD = (
            self._activation_harvester.sequence_length,
            self._activation_harvester.n_models,
            self._activation_harvester.n_layers,
            self._activation_harvester.d_model,
        )

    def _get_activations_iterator_SMLD(self) -> Iterator[torch.Tensor]:
        for activations_BSMLD in self._activation_harvester.get_activations_iterator_BSMLD():
            yield from activations_BSMLD
            # If this "yield from" is hard to understand, it is equivalent to:
            # ```
            # for activations_SMLD in activations_BSMLD:
            #     yield activations_SMLD
            # ```

    def get_shuffled_activations_iterator_BSMLD(self) -> Iterator[torch.Tensor]:
        if self._batch_size > self._shuffle_buffer_size // 2:
            raise ValueError(
                f"Batch size cannot be greater than half the buffer size, {self._batch_size} > {self._shuffle_buffer_size // 2}"
            )

        buffer_BSMLD = torch.empty((self._shuffle_buffer_size, *self._activation_shape_SMLD))

        available_indices = set()
        stale_indices = set(range(self._shuffle_buffer_size))

        iterator_SMLD = self._get_activations_iterator_SMLD()

        while True:
            # refill buffer
            for stale_idx, activation_SMLD in zip(list(stale_indices), iterator_SMLD, strict=False):
                assert activation_SMLD.shape == self._activation_shape_SMLD, (
                    f"activation_SMLD.shape should be {self._activation_shape_SMLD} but was {activation_SMLD.shape}"
                )
                buffer_BSMLD[stale_idx] = activation_SMLD
                available_indices.add(stale_idx)
                stale_indices.remove(stale_idx)

            # yield batches until buffer is half empty
            while len(available_indices) >= self._shuffle_buffer_size // 2:
                batch_indices = random.sample(list(available_indices), self._batch_size)
                yield buffer_BSMLD[batch_indices]
                available_indices.difference_update(batch_indices)
                stale_indices.update(batch_indices)
