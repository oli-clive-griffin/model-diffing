from datasets import load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.token_loader.base import TokenSequenceLoader
from model_diffing.data.token_loader.connor import ConnorGemma2TokenSequenceLoader
from model_diffing.data.token_loader.huggingface import HuggingfaceTextDatasetTokenSequenceLoader
from model_diffing.data.token_loader.math import MathDatasetTokenSequenceLoader
from model_diffing.data.token_loader.toy import ToyOverfittingTokenSequenceLoader
from model_diffing.scripts.config_common import (
    ConnorGemma2Config,
    HuggingfaceTextDatasetConfig,
    MathDatasetConfig,
    SequenceIteratorCfg,
    ToyOverfittingConfig,
)


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
                include_base_answers=cfg.include_base_answers,
                include_reasoning_answers=cfg.include_reasoning_answers,
            )
    raise ValueError(f"Unknown tokens sequence iterator: {cfg}")

# import torch
# from typing import Any, Dict, List, cast
# from datasets import Dataset

# # Create a mock dataset
# conversations = []
# for i in range(10):  # 10 examples
#     base_conv = [
#         {"role": "user", "content": f"Question {i}"},
#         {"role": "assistant", "content": f"Base answer {i}"}
#     ]
#     thinking_conv = [
#         {"role": "user", "content": f"Question {i}"},
#         {"role": "assistant", "content": f"Reasoning answer {i}"}
#     ]
#     conversations.append({
#         "messages": base_conv,
#         "reannotated_messages": thinking_conv
#     })

# # if __name__ == "__main__":

# # # Create a dataset
# # dataset = Dataset.from_list(conversations)

# # # Mock tokenizer
# # class MockTokenizer:
# #     def __call__(self, texts, return_tensors="pt", padding=None, padding_side=None, max_length=None):
# #         # Simply create fake token IDs for demonstration
# #         max_len = max_length or 5
# #         input_ids = torch.zeros((len(texts), max_len), dtype=torch.long)
# #         attention_mask = torch.ones((len(texts), max_len), dtype=torch.bool)
        
# #         # Fill with some dummy values
# #         for i, text in enumerate(texts):
# #             # Use the length of the text as a feature for demonstration
# #             input_ids[i, :min(len(text), max_len)] = len(text) % 10 + 1
        
# #         return {"input_ids": input_ids, "attention_mask": attention_mask}

# # # Create instance variables
# # tokenizer = MockTokenizer()
# # batch_size = 4  # Original batch size
# # max_sequence_length = 10
# # special_ids = torch.tensor([0])  # Special token IDs

# # # The map function
# # def map_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
# #     batch_conversations = batch["reannotated_messages"]
# #     batch_base_convs = batch["messages"]
# #     sequences: List[str] = []
    
# #     for base_conversation, thinking_conversation in zip(batch_base_convs, batch_conversations):
# #         user_question = thinking_conversation[0]
# #         assert user_question["role"] == "user"
# #         question = user_question["content"]
        
# #         base_response = base_conversation[1]
# #         assert base_response["role"] == "assistant"
# #         base_answer = base_response["content"]
        
# #         thinking_response = thinking_conversation[1]
# #         assert thinking_response["role"] == "assistant"
# #         reasoning = thinking_response["content"]
        
# #         sequences.append(question + base_answer)
# #         sequences.append(question + reasoning)

# #     tok_res = tokenizer(
# #         sequences,
# #         return_tensors="pt",
# #         padding="max_length",
# #         padding_side="left",
# #         max_length=max_sequence_length,
# #     )
    
# #     batch["tokens_BS"] = cast(torch.Tensor, tok_res["input_ids"])
# #     batch["attention_mask_BS"] = cast(torch.Tensor, tok_res["attention_mask"]).bool()
# #     batch["special_tokens_mask_BS"] = torch.isin(batch["tokens_BS"], special_ids)
    
# #     return batch

# # # Using Option 1: Reduce the batch size in the initial map call
# # tokens_dataset = (
# #     dataset.map(map_fn, batched=True, batch_size=batch_size // 2)  # Use half the batch size here
# #     .select_columns(["tokens_BS", "attention_mask_BS", "special_tokens_mask_BS"])
# #     .with_format("torch")
# # )

# # # Print the shapes to verify
# # for batch_idx, batch in enumerate(tokens_dataset.iter(batch_size=batch_size)):
# #     print(f"Batch {batch_idx}:")
# #     print(f"  tokens_BS shape: {batch['tokens_BS'].shape}")
# #     print(f"  attention_mask_BS shape: {batch['attention_mask_BS'].shape}")
# #     print(f"  special_tokens_mask_BS shape: {batch['special_tokens_mask_BS'].shape}")
# #     print(f"  Expected batch size: {batch_size}")
# #     if batch_idx >= 2:  # Just show a few batches
# #         break