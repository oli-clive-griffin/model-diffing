from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from crosscoding.data.shuffle import batch_shuffle_tensor_iterator_BX


@dataclass
class TokensSequenceBatch:
    tokens_HS: torch.Tensor
    special_tokens_mask_HS: torch.Tensor

    def __post_init__(self):
        if self.special_tokens_mask_HS.sum() / self.special_tokens_mask_HS.numel() > 0.1:
            pass
            # logger.warning("more than 10% of tokens are special tokens, this is unexpected")

        if self.tokens_HS.dtype != torch.long:
            raise ValueError(f"tokens_HS should be a long tensor, got {self.tokens_HS.dtype}")

        if self.special_tokens_mask_HS.dtype != torch.bool:
            raise ValueError(
                f"special_tokens_mask_HS should be a boolean tensor, got {self.special_tokens_mask_HS.dtype}"
            )


class TokenSequenceLoader:
    def __init__(
        self,
        hf_dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        batch_size: int,
        shuffle_buffer_size: int | None = None,
        cache_dir: str | None = None,
    ):
        self._hf_dataset = load_dataset(
            path=hf_dataset_name,
            streaming=True,
            cache_dir=cache_dir,
            split="train",
        )

        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

    def _get_sequence_iterator_S(self) -> Iterator[torch.Tensor]:
        example_S = torch.empty(self._sequence_length, dtype=torch.long)
        example_pointer = 0

        for example in self._hf_dataset:
            text = cast(dict[str, Any], example)["text"]
            tokens = self._tokenizer(text, return_tensors="pt")["input_ids"]
            tokens = cast(torch.Tensor, tokens)
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
    def _get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        # we shuffle this iterator (only between, not within, sequences) so that we don't have to worry
        # about long documents introducing high feature correlations
        # this shuffler returns batches of sequences of tokens.
        special_ids = torch.tensor(self._tokenizer.all_special_ids)
        if self._shuffle_buffer_size is not None:
            for tokens_HS in batch_shuffle_tensor_iterator_BX(
                tensor_iterator_X=self._get_sequence_iterator_S(),
                shuffle_buffer_size=self._shuffle_buffer_size,
                yield_batch_size_B=self._batch_size,
            ):
                special_tokens_mask_HS = torch.isin(tokens_HS, special_ids)
                yield TokensSequenceBatch(
                    tokens_HS=tokens_HS,
                    special_tokens_mask_HS=special_tokens_mask_HS,
                )
        else:
            iterator_S = self._get_sequence_iterator_S()
            out_tokens_S: list[torch.Tensor] = []
            for sample_S in iterator_S:
                if len(out_tokens_S) == self._batch_size:
                    tokens_HS = torch.stack(out_tokens_S)
                    yield TokensSequenceBatch(
                        tokens_HS=tokens_HS,
                        special_tokens_mask_HS=torch.isin(tokens_HS, special_ids),
                    )
                    out_tokens_S.clear()
                out_tokens_S.append(sample_S)

    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        return self._get_sequences_batch_iterator
