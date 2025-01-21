from collections.abc import Iterator

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.dataloader.activations import (
    ActivationsHarvester,
    TokensActivationsShuffler,
    iterate_over_tokens,
)
from model_diffing.dataloader.token_loader import (
    CommonCorpusTokenSequenceIterator,
    ConnorGemma2TokenSequenceLoader,
    TokenSequenceLoader,
)
from model_diffing.scripts.config_common import (
    ActivationsIteratorConfig,
    DataConfig,
    SequenceTokensIteratorConfig,
)


def build_dataloader_BMLD(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    cache_dir: str,
) -> Iterator[torch.Tensor]:
    acts_iterator = _build_activations_iterator(cfg.activations_iterator, cache_dir, llms)

    shuffler = TokensActivationsShuffler(
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        activations_iterator_BSMLD=acts_iterator.get_activations_iterator_BSMLD(),
        activations_reshaper=iterate_over_tokens,
        batch_size=cfg.batch_size,
    )

    return shuffler.get_shuffled_activations_iterator()


def _build_activations_iterator(
    cfg: ActivationsIteratorConfig,
    cache_dir: str,
    llms: list[HookedTransformer],
) -> ActivationsHarvester:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")
    sequence_tokens_iterator = _build_tokens_sequence_iterator(cfg.sequence_tokens_iterator, cache_dir, tokenizer)
    return ActivationsHarvester(
        llms=llms,
        batch_size=cfg.harvest_batch_size,
        layer_indices_to_harvest=cfg.layer_indices_to_harvest,
        sequence_tokens_iterator=sequence_tokens_iterator.get_sequence_iterator(),
    )


def _build_tokens_sequence_iterator(
    cfg: SequenceTokensIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenSequenceLoader:
    if cfg.name == "common_corpus":
        if cfg.sequence_length is None:
            raise ValueError("sequence_length must be provided for common_corpus")
        return CommonCorpusTokenSequenceIterator(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            sequence_length=cfg.sequence_length,
        )
    elif cfg.name == "connor_gemma":
        return ConnorGemma2TokenSequenceLoader(
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown tokens sequence iterator config name: {cfg}")
