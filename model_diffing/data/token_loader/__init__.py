from typing import Literal

from datasets import load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.token_loader.base import TokenSequenceLoader
from model_diffing.data.token_loader.connor import ConnorGemma2TokenSequenceLoader
from model_diffing.data.token_loader.huggingface import HuggingfaceTextDatasetTokenSequenceLoader
from model_diffing.data.token_loader.math import MathDatasetTokenSequenceLoader
from model_diffing.data.token_loader.toy import ToyOverfittingTokenSequenceLoader
from model_diffing.utils import BaseModel


class HuggingfaceTextDatasetConfig(BaseModel):
    type: Literal["HuggingfaceTextDatasetTokenSequenceLoader"] = "HuggingfaceTextDatasetTokenSequenceLoader"
    hf_dataset_name: str
    sequence_length: int
    shuffle_buffer_size: int | None = None


class ConnorGemma2Config(BaseModel):
    type: Literal["ConnorGemma2TokenSequenceLoader"] = "ConnorGemma2TokenSequenceLoader"
    # No additional parameters needed


class ToyOverfittingConfig(BaseModel):
    type: Literal["ToyOverfittingTokenSequenceLoader"] = "ToyOverfittingTokenSequenceLoader"
    sequence_length: int
    vocab_size: int = 10
    first_n_tokens_special: int = 2


class MathDatasetConfig(BaseModel):
    type: Literal["MathDatasetTokenSequenceLoader"] = "MathDatasetTokenSequenceLoader"
    max_sequence_length: int
    include_base_answers: bool = False
    include_reasoning_answers: bool = False


TokenSequenceLoaderCfg = HuggingfaceTextDatasetConfig | ConnorGemma2Config | ToyOverfittingConfig | MathDatasetConfig


def default_tokens_sequence_iterator():
    return HuggingfaceTextDatasetConfig(
        hf_dataset_name="monology/pile-uncopyrighted",
        sequence_length=2048,
        shuffle_buffer_size=None,
    )


def build_tokens_sequence_loader(
    cfg: TokenSequenceLoaderCfg,
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
