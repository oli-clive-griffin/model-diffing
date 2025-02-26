# %%
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.log import logger
from model_diffing.scripts.config_common import (
    ConnorGemma2Config,
    HuggingfaceTextDatasetConfig,
    MathDatasetConfig,
    SequenceIteratorCfg,
    ToyOverfittingConfig,
)


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
        for tokens_BS in batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=self._get_sequence_iterator_S(),
            shuffle_buffer_size=self._shuffle_buffer_size,
            yield_batch_size=self._batch_size,
            name="token sequence loader",
        ):
            special_tokens_mask_BS = torch.isin(tokens_BS, special_ids)
            yield TokensSequenceBatch(
                tokens_BS=tokens_BS,
                special_tokens_mask_BS=special_tokens_mask_BS,
            )

    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        return self._get_sequences_batch_iterator

    def num_batches(self) -> int | None:
        # This kind of can't easily be computed, because it's a function of sequence length and each example's length
        # This is a good example of why `num_batches` is `None`able
        return None


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
            tokens_BS = torch.randint(
                0,
                self._vocab_size,
                (self._batch_size, self._sequence_length),
                generator=self._generator,
            )
            special_tokens_mask_BS = tokens_BS < self._first_n_tokens_special

            yield TokensSequenceBatch(
                tokens_BS=tokens_BS,
                special_tokens_mask_BS=special_tokens_mask_BS,
            )

    def num_batches(self) -> int | None:
        return None


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
            logger.warning(f"special token exclusion not yet implemented for {self.__name__}")
            yield TokensSequenceBatch(
                tokens_BS=torch.tensor(tokens_S),
                special_tokens_mask_BS=torch.zeros(len(tokens_S), dtype=torch.bool),
            )

    def num_batches(self) -> int | None:
        return self.N_ROWS // self._batch_size


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
                tokens_BS=tokens_1S,
                special_tokens_mask_BS=torch.zeros_like(tokens_1S, dtype=torch.bool),
            )


class MathDatasetTokenSequenceLoader(TokenSequenceLoader):
    # https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT

    DATASET_PATH = "ServiceNow-AI/R1-Distill-SFT"
    DATASET_NAME = "v1"
    SPLIT = "train"  # there's only a train split

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_sequence_length: int,
        cache_dir: str,
        base_answers: bool = False,
        reasoning_answers: bool = False,
    ):
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_sequence_length = max_sequence_length
        self._base_answers = base_answers
        self._reasoning_answers = reasoning_answers
        self._base_ds = cast(
            IterableDataset,
            load_dataset(
                self.DATASET_PATH,
                self.DATASET_NAME,
                split=self.SPLIT,
                streaming=True,
                cache_dir=cache_dir,
            ),
        )

    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        special_ids = torch.tensor(self._tokenizer.all_special_ids)

        def map_fn(batch: dict[str, Any]) -> dict[str, Any]:
            batch_conversations = batch["reannotated_messages"]  # list(batch) of list(user, assistant) of message dicts
            batch_base_convs = batch["messages"]

            sequences: list[str] = []
            for base_conversation, thinking_conversation in zip(batch_base_convs, batch_conversations, strict=True):
                user_question = thinking_conversation[0]
                assert user_question["role"] == "user"
                question = user_question["content"]

                base_response = base_conversation[1]
                assert base_response["role"] == "assistant"
                base_answer = base_response["content"]

                thinking_response = thinking_conversation[1]
                assert thinking_response["role"] == "assistant"
                reasoning = thinking_response["content"]

                if self._base_answers:
                    sequences.append(question + base_answer)
                elif self._reasoning_answers:
                    sequences.append(question + reasoning)
                else:
                    sequences.append(question)

            tok_res = self._tokenizer.__call__(
                sequences,
                return_tensors="pt",
                padding="max_length",
                padding_side="left",  # type: ignore # this type is just plain wrong for some reason
                max_length=self._max_sequence_length,
            )

            batch["tokens_BS"] = cast(torch.Tensor, tok_res["input_ids"])
            batch["attention_mask_BS"] = cast(torch.Tensor, tok_res["attention_mask"]).bool()
            batch["special_tokens_mask_BS"] = torch.isin(batch["tokens_BS"], special_ids)
            return batch

        tokens_dataset = (
            self._base_ds.map(map_fn, batched=True, batch_size=self._batch_size)
            .select_columns(["tokens_BS", "attention_mask_BS", "special_tokens_mask_BS"])
            .with_format(type="torch")
            .batch(
                batch_size=self._batch_size * 2
                if (self._base_answers and self._reasoning_answers)
                else self._batch_size
            )
        )

        for batch in tokens_dataset:
            batch = cast(dict[str, torch.Tensor], batch)
            yield TokensSequenceBatch(
                tokens_BS=batch["tokens_BS"],
                special_tokens_mask_BS=batch["special_tokens_mask_BS"],
                attention_mask_BS=batch["attention_mask_BS"],
            )

    def num_batches(self) -> int | None:
        return None


def build_tokens_sequence_loader(
    cfg: SequenceIteratorCfg,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> TokenSequenceLoader:
    match cfg:
        case HuggingfaceTextDatasetConfig():
            text_dataset = load_dataset(
                path=cfg.hf_dataset_name,
                streaming=True,
                cache_dir=cache_dir,
                split="train",
            )
            return HuggingfaceTextDatasetTokenSequenceLoader(
                cache_dir=cache_dir,
                tokenizer=tokenizer,
                batch_size=batch_size,
                hf_dataset=text_dataset,
                sequence_length=cfg.sequence_length,
                shuffle_buffer_size=cfg.shuffle_buffer_size,
            )
        case ConnorGemma2Config():
            return ConnorGemma2TokenSequenceLoader(
                cache_dir=cache_dir,
                batch_size=batch_size,
            )
        case ToyOverfittingConfig():
            return ToyOverfittingTokenSequenceLoader(
                batch_size=batch_size,
                sequence_length=cfg.sequence_length,
                vocab_size=cfg.vocab_size,
                first_n_tokens_special=cfg.first_n_tokens_special,
            )
        case MathDatasetConfig():
            return MathDatasetTokenSequenceLoader(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_sequence_length=cfg.max_sequence_length,
                cache_dir=cache_dir,
            )
    raise ValueError(f"Unknown tokens sequence iterator: {cfg}")


# %%

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    sl = MathDatasetTokenSequenceLoader(
        tokenizer=tok,
        batch_size=1,
        max_sequence_length=1024,
        cache_dir=".cache",
    )

    for seqh in sl.get_sequences_batch_iterator():
        breakpoint()
