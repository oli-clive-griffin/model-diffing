from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import IterableDataset, load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.token_loader.base import TokenSequenceLoader, TokensSequenceBatch


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
