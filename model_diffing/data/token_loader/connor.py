from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore

from model_diffing.data.token_loader.base import TokenSequenceLoader, TokensSequenceBatch
from model_diffing.log import logger


class ConnorGemma2TokenSequenceLoader(TokenSequenceLoader):
    # these properties are simply taken from https://huggingface.co/datasets/ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2
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

    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        dataset = load_dataset(
            self.HF_TOKENISED_DATASET,
            streaming=True,
            cache_dir=self._cache_dir,
            split="train",
            batch_size=self._batch_size,
        )

        for example in cast(Iterator[dict[str, Any]], dataset):
            if not isinstance(example["input_ids"], list):
                logger.warning(f"my assumption was wrong, expected list but got {type(example['input_ids'])}")

            tokens_S = example["input_ids"]
            logger.warning(f"special token exclusion not yet implemented for {self.__class__.__name__}")
            yield TokensSequenceBatch(
                tokens_HS=torch.tensor(tokens_S),
                special_tokens_mask_HS=torch.zeros(len(tokens_S), dtype=torch.bool),
            )

    def num_batches(self) -> int | None:
        return self.N_ROWS // self._batch_size
