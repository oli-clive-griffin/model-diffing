from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.dataloader.shuffle import batch_shuffle_tensor_iterator_BX


class TokenSequenceLoader(ABC):
    @abstractmethod
    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...  # not using __len__ because __len__ doesn't work well with `| None`


class CommonCorpusTokenSequenceLoader(TokenSequenceLoader):
    COMMON_CORPUS_HF_DATASET = "PleIAs/common_corpus"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        shuffle_buffer_size: int,
        batch_size: int,
        cache_dir: str | None = None,
    ):
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

    def _get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        text_dataset = load_dataset(
            self.COMMON_CORPUS_HF_DATASET, streaming=True, cache_dir=self._cache_dir, split="train"
        )

        for example in text_dataset:
            example = cast(dict[str, Any], example)
            tokeniser_result = self._tokenizer(example["text"])
            seq_tokens_S = torch.tensor(tokeniser_result["input_ids"])
            assert len(seq_tokens_S.shape) == 1, f"seq_tokens_S.shape should be 1D but was {seq_tokens_S.shape}"
            num_full_sequences = len(seq_tokens_S) // self._sequence_length
            if num_full_sequences == 0:
                continue

            for i in range(0, num_full_sequences * self._sequence_length, self._sequence_length):
                yield seq_tokens_S[i : i + self._sequence_length]

    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        # then, shuffle this iterator (only between, not within, sequences) so that we don't have to worry
        # about long documents introducing high feature correlations
        # this shuffler returns batches, hence (B, S)
        return batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=self._get_sequence_iterator(),
            shuffle_buffer_size=self._shuffle_buffer_size,
            yield_batch_size=self._batch_size,
        )

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


# For example, we could do:
# class LocalDatasetTokenSequenceIterator(TokenSequenceIterator):
#     """backed by a file"""
#     ...

# or

# class MemoryTokenSequenceIterator(TokenSequenceLoader):
#     def __init__(self, tokens_AS: torch.Tensor):
#         self.tokens_AS = tokens_AS

#     def __iter__(self) -> Iterator[torch.Tensor]:
#         return iter(self.tokens_AS)

# from itertools import islice

# class ThePileTokenSequenceIterator(TokenSequenceLoader):
#     HF_TOKENISED_DATASET = "EleutherAI/pile"

#     def __init__(self, cache_dir: str):
#         self._cache_dir = cache_dir

#     def get_sequence_iterator(self) -> Iterator[torch.Tensor]:
#         dataset = load_dataset(self.HF_TOKENISED_DATASET, streaming=True, cache_dir=self._cache_dir, split="train")
#         for example in dataset:
#             tokens = torch.tensor(example["input_ids"])  # type: ignore
#             yield tokens


if __name__ == "__main__":
    from itertools import islice

    token_loader = ConnorGemma2TokenSequenceLoader(cache_dir=".cache", batch_size=16)
    for tokens in islice(token_loader.get_sequences_batch_iterator(), 10):
        print(tokens.shape)
