from typing import cast

import torch
from datasets import IterableDataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from crosscoding.data.token_loader import HuggingfaceTextDatasetTokenSequenceLoader


def test_huggingface_text_dataset_token_sequence_loader():
    def make_iterator():
        i = 0
        while True:
            yield {"text": f"hello world {i}"}
            i += 1

    mock_dataset = cast(IterableDataset, make_iterator())

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160M",
        add_bos_token=True,  # HookedTransformer adds this when finding the tokenizer
    )
    assert tokenizer.special_tokens_map["bos_token"] == "<|endoftext|>"  # just a sanity check

    sequence_length = 7  # to test bos wrapping

    token_sequence_loader = HuggingfaceTextDatasetTokenSequenceLoader(
        hf_dataset=mock_dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        shuffle_buffer_size=1,
        batch_size=1,
    )

    example_tokens = next(token_sequence_loader.get_sequences_batch_iterator())
    assert example_tokens.tokens_HS.shape == (1, sequence_length)
    example_text = tokenizer.decode(example_tokens.tokens_HS[0])
    assert example_text == "<|endoftext|>hello world 0<|endoftext|>hello world"

    assert example_tokens.special_tokens_mask_HS.sum() == 2
    assert torch.all(example_tokens.special_tokens_mask_HS[0] == torch.tensor([1, 0, 0, 0, 1, 0, 0], dtype=torch.bool))

    second_example_tokens = next(token_sequence_loader.get_sequences_batch_iterator())
    assert second_example_tokens.tokens_HS.shape == (1, sequence_length)
    second_example_text = tokenizer.decode(second_example_tokens.tokens_HS[0])
    assert second_example_text == " 1<|endoftext|>hello world 2<|endoftext|>hello"
    assert torch.all(
        second_example_tokens.special_tokens_mask_HS[0] == torch.tensor([0, 1, 0, 0, 0, 1, 0], dtype=torch.bool)
    )
