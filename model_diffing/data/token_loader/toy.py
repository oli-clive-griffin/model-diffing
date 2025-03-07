from collections.abc import Iterator
from typing import cast

import torch
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.token_loader.base import TokenSequenceLoader, TokensSequenceBatch


class ToyOverfittingTokenSequenceLoader(TokenSequenceLoader):
    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        vocab_size: int = 10,
        first_n_tokens_special: int = 2,
    ):
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._vocab_size = vocab_size
        self._first_n_tokens_special = first_n_tokens_special
        self._generator = torch.Generator().manual_seed(0)

    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        while True:
            tokens_HS = torch.randint(
                0,
                self._vocab_size,
                (self._batch_size, self._sequence_length),
                generator=self._generator,
            )
            special_tokens_mask_HS = tokens_HS < self._first_n_tokens_special

            yield TokensSequenceBatch(
                tokens_HS=tokens_HS,
                special_tokens_mask_HS=special_tokens_mask_HS,
            )

    def num_batches(self) -> int | None:
        return None


class MemoryTokenSequenceLoader(TokenSequenceLoader):
    BATCH_SIZE = 1

    def __init__(self, sequences: list[str], tokenizer: PreTrainedTokenizerBase):
        self._sequences = sequences
        self._tokenizer = tokenizer

    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        for seq in self._sequences:
            tokens_1S = self._tokenizer(seq, return_tensors="pt")["input_ids"]
            tokens_1S = cast(torch.Tensor, tokens_1S)
            assert len(tokens_1S.shape) == 2, f"tokens.shape should be 2D but was {tokens_1S.shape}"
            assert tokens_1S.shape[0] == 1, f"tokens.shape should have a batch dimension of 1 but was {tokens_1S.shape}"
            yield TokensSequenceBatch(
                tokens_HS=tokens_1S,
                special_tokens_mask_HS=torch.zeros_like(tokens_1S, dtype=torch.bool),
            )
