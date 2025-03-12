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
        include_base_answers: bool = False,
        include_reasoning_answers: bool = False,
    ):
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_sequence_length = max_sequence_length
        self._include_base_answers = include_base_answers
        self._include_reasoning_answers = include_reasoning_answers
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

        def tensorize_batch(batch: dict[str, Any]) -> dict[str, Any]:
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

                match self._include_base_answers, self._include_reasoning_answers:
                    case (True, True):
                        sequences.append(question + base_answer)
                        sequences.append(question + reasoning)
                    case (True, False):
                        sequences.append(question + base_answer)
                    case (False, True):
                        sequences.append(question + reasoning)
                    case (False, False):
                        sequences.append(question)
                    case _:
                        raise ValueError(
                            f"Invalid combination of base_answers and reasoning_answers: {self._include_base_answers=}, {self._include_reasoning_answers=}"
                        )

            tok_res = self._tokenizer.__call__(
                sequences,
                return_tensors="pt",
                padding="longest",
                padding_side="right",  # type: ignore # this type is just plain wrong for some reason
                max_length=self._max_sequence_length,
            )

            return {  # TODO: figure out why this is needed. Sequences are coming out of the tokenizer with length > max_sequence_length
                "tokens_HS": cast(torch.Tensor, tok_res["input_ids"])[:, : self._max_sequence_length],
                "special_tokens_mask_HS": torch.isin(
                    cast(torch.Tensor, tok_res["input_ids"])[:, : self._max_sequence_length], special_ids
                ),
            }

        assert self._batch_size % 2 == 0
        will_double_example_count = self._include_base_answers and self._include_reasoning_answers
        tensorize_batch_size = self._batch_size // 2 if will_double_example_count else self._batch_size

        tokens_dataset = (
            self._base_ds.map(
                tensorize_batch,
                batched=True,
                batch_size=tensorize_batch_size,
                remove_columns=self._base_ds.column_names,
            )
            .with_format(type="torch")
            .batch(batch_size=self._batch_size)
        )

        for batch in tokens_dataset:
            batch = cast(dict[str, torch.Tensor], batch)
            assert batch["tokens_HS"].shape == batch["special_tokens_mask_HS"].shape
            yield TokensSequenceBatch(
                tokens_HS=batch["tokens_HS"],
                special_tokens_mask_HS=batch["special_tokens_mask_HS"],
            )

    def num_batches(self) -> int | None:
        return None


if __name__ == "__main__":
    from itertools import islice

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    loader = MathDatasetTokenSequenceLoader(
        tokenizer,
        4,
        2048,
        ".cache",
        include_base_answers=True,
        include_reasoning_answers=True,
    )
    for batch in islice(loader.get_sequences_batch_iterator(), 10):
        print()
        print(batch.tokens_HS.shape)
        print()
        for seq in batch.tokens_HS.unbind(0):
            print(
                tokenizer.decode(seq),
            )
