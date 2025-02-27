from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass
class TokensSequenceBatch:
    tokens_BS: torch.Tensor
    special_tokens_mask_BS: torch.Tensor
    attention_mask_BS: torch.Tensor | None = None

    def __post_init__(self):
        if self.special_tokens_mask_BS.sum() / self.special_tokens_mask_BS.numel() > 0.1:
            pass
            # logger.warning("more than 10% of tokens are special tokens, this is unexpected")

        if self.tokens_BS.dtype != torch.long:
            raise ValueError(f"tokens_BS should be a long tensor, got {self.tokens_BS.dtype}")

        if self.special_tokens_mask_BS.dtype != torch.bool:
            raise ValueError(
                f"special_tokens_mask_BS should be a boolean tensor, got {self.special_tokens_mask_BS.dtype}"
            )

        if self.attention_mask_BS is not None:
            # all attention masks positions should also be special tokens
            if self.attention_mask_BS.dtype != torch.bool:
                raise ValueError(f"attention_mask_BS should be a boolean tensor, got {self.attention_mask_BS.dtype}")

            attn_ignored_tokens = ~self.attention_mask_BS
            if (attn_ignored_tokens & self.special_tokens_mask_BS).sum() != attn_ignored_tokens.sum():
                breakpoint()
                raise ValueError("all special tokens positions should also be in the attention mask")


class TokenSequenceLoader(ABC):
    @abstractmethod
    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]: ...

    def num_batches(self) -> int | None:  # not using __len__ because __len__ doesn't work well with `| None`
        return None
