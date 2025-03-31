from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import torch


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


class TokenSequenceLoader(ABC):
    @abstractmethod
    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]: ...
