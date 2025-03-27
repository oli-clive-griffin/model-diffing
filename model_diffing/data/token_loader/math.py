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
        raise ValueError("TOKENIZATION IS NOT WORKING WITH <think> TAGS")

    def _format_question(self, question: str, answer: str | None = None) -> str:
        out = f"User: {question}\n"
        if answer is not None:
            out += f"Assistant: {answer}\n"
        return out

    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        special_ids = torch.tensor(self._tokenizer.all_special_ids)

        def tensorize_batch(batch: dict[str, Any]) -> dict[str, Any]:
            batch_base_convs = batch["messages"]
            batch_thinking_convs = batch[
                "reannotated_messages"
            ]  # list(batch) of list(user, assistant) of message dicts

            sequences: list[str] = []
            for base_conversation, thinking_conversation in zip(batch_base_convs, batch_thinking_convs, strict=True):
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
                        sequences.append(self._format_question(question, base_answer))
                        sequences.append(self._format_question(question, reasoning))
                    case (True, False):
                        sequences.append(self._format_question(question, base_answer))
                    case (False, True):
                        sequences.append(self._format_question(question, reasoning))
                    case (False, False):
                        sequences.append(self._format_question(question))
                    case _:
                        raise ValueError(
                            f"Invalid combination of base_answers and reasoning_answers: {self._include_base_answers=}, {self._include_reasoning_answers=}"
                        )

            tok_res = self._tokenizer(
                sequences,
                return_tensors="pt",
                padding="longest",
                padding_side="right",  # type: ignore # this type is just plain wrong for some reason
                max_length=self._max_sequence_length,
                truncation=True,
            )
            seq = cast(torch.Tensor, tok_res["input_ids"])
            return {"tokens_HS": seq, "special_tokens_mask_HS": torch.isin(seq, special_ids)}

        tensorize_batch_size = self._batch_size

        will_double_example_count = self._include_base_answers and self._include_reasoning_answers
        if will_double_example_count:
            assert tensorize_batch_size % 2 == 0
            tensorize_batch_size //= 2

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
    from tqdm import tqdm  # type: ignore
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
    pbar = tqdm(desc="tokens processed", unit="tokens", total=None)
    for batch in loader.get_sequences_batch_iterator():
        pbar.update((~batch.special_tokens_mask_HS).sum().item())
        print(batch.tokens_HS.shape)
