from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from typing import Any, cast

import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.scripts.config_common import SequenceIteratorConfig


class TokenSequenceLoader(ABC):
    @abstractmethod
    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...  # not using __len__ because __len__ doesn't work well with `| None`


COMMON_CORPUS_HF_DATASET = "PleIAs/common_corpus"
THE_PILE_UNCOPYRIGHTED_HF_DATASET = "monology/pile-uncopyrighted"


class HuggingfaceTextDatasetTokenSequenceLoader(TokenSequenceLoader):
    def __init__(
        self,
        hf_dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        shuffle_buffer_size: int,
        batch_size: int,
        cache_dir: str | None = None,
    ):
        self._hf_dataset = hf_dataset
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

    def _get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        example_S = torch.empty(self._sequence_length, dtype=torch.long)
        example_pointer = 0

        for example in self._hf_dataset:
            tokens = cast(
                torch.Tensor,
                self._tokenizer(
                    cast(dict[str, Any], example)["text"],
                    return_tensors="pt",
                )["input_ids"],
            )
            assert len(tokens.shape) == 2, f"tokens.shape should be 2D but was {tokens.shape}"
            assert tokens.shape[0] == 1, f"tokens.shape should have a batch dimension of 1 but was {tokens.shape}"

            seq_tokens_S = tokens.squeeze(0)
            seq_pointer = 0

            while seq_pointer < seq_tokens_S.shape[0]:
                tokens_left_to_fill_example = example_S.shape[0] - example_pointer
                tokens_left_in_seq = seq_tokens_S.shape[0] - seq_pointer

                tokens_to_copy = min(tokens_left_to_fill_example, tokens_left_in_seq)

                example_S[example_pointer : example_pointer + tokens_to_copy] = (  #
                    seq_tokens_S[seq_pointer : seq_pointer + tokens_to_copy]
                )

                # this is always valid because of the `min` above
                example_pointer += tokens_to_copy
                seq_pointer += tokens_to_copy

                if example_pointer == self._sequence_length:
                    example_pointer = 0
                    yield example_S

    @cached_property
    def _get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        # we shuffle this iterator (only between, not within, sequences) so that we don't have to worry
        # about long documents introducing high feature correlations
        # this shuffler returns batches of sequences of tokens.
        return batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=self._get_sequence_iterator(),
            shuffle_buffer_size=self._shuffle_buffer_size,
            yield_batch_size=self._batch_size,
            name="token sequence loader",
        )

    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        return self._get_sequences_batch_iterator

    def num_batches(self) -> int | None:
        # This kind of can't easily be computed, because it's a function of sequence length and each example's length
        # This is a good example of why `num_batches` is `None`able
        return None


class ToyOverfittingTokenSequenceLoader(TokenSequenceLoader):
    def __init__(self, batch_size: int, sequence_length: int):
        self._batch_size = batch_size
        self._sequence_length = sequence_length

    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.randint(0, 1000, (self._batch_size, self._sequence_length))

    def num_batches(self) -> int | None:
        return None


class ConnorGemma2TokenSequenceLoader(TokenSequenceLoader):
    HF_TOKENISED_DATASET = "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2"
    SEQUENCE_LENGTH = 1024
    N_ROWS = 963_566

    def __init__(self, cache_dir: str, batch_size: int):
        """expects a tokenised huggingface dataset"""
        self._cache_dir = cache_dir
        self._batch_size = batch_size

    def _batch_accumulator(self, sequence_iterator: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
        """accumulate sequences into batches, yielding batches of shape (B, S)"""
        buffer = torch.empty((self._batch_size, self.SEQUENCE_LENGTH))
        pos = 0

        for sequence in sequence_iterator:
            buffer[pos] = sequence
            pos += 1
            if pos == self._batch_size:
                yield buffer.clone()
                pos = 0

    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        dataset = load_dataset(
            self.HF_TOKENISED_DATASET,
            streaming=True,
            cache_dir=self._cache_dir,
            split="train",
            batch_size=self._batch_size,
        )
        sequence_iterator = (
            torch.tensor(tokens_S["input_ids"]) for tokens_S in cast(Iterator[dict[str, Any]], dataset)
        )
        return self._batch_accumulator(sequence_iterator)

    def num_batches(self) -> int | None:
        return self.N_ROWS // self._batch_size


def build_tokens_sequence_loader(
    cfg: SequenceIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> TokenSequenceLoader:
    if cfg.classname == "HuggingfaceTextDatasetTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided")
        text_dataset = load_dataset(
            path=cfg.kwargs["hf_dataset_name"],
            streaming=True,
            cache_dir=cache_dir,
            split="train",
        )
        return HuggingfaceTextDatasetTokenSequenceLoader(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            batch_size=batch_size,
            hf_dataset=text_dataset,
            sequence_length=cfg.kwargs["sequence_length"],
            shuffle_buffer_size=cfg.kwargs["shuffle_buffer_size"],
        )
    elif cfg.classname == "ConnorGemma2TokenSequenceLoader":
        return ConnorGemma2TokenSequenceLoader(
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
    elif cfg.classname == "ToyOverfittingTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided")
        return ToyOverfittingTokenSequenceLoader(
            batch_size=batch_size,
            **cfg.kwargs,
        )

    raise ValueError(f"Unknown tokens sequence iterator config name: {cfg}")
