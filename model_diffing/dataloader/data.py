from collections.abc import Iterator

import torch
from transformers import PreTrainedTokenizerBase

from model_diffing.dataloader.activations import ActivationsHarvester
from model_diffing.dataloader.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.dataloader.token_loader import (
    CommonCorpusTokenSequenceIterator,
    ConnorGemma2TokenSequenceLoader,
    TokenSequenceLoader,
    ToyOverfittingTokenSequenceIterator,
)
from model_diffing.scripts.config_common import DataConfig, SequenceIteratorConfig
from model_diffing.scripts.llms import build_llms


def build_dataloader_BMLD(
    cfg: DataConfig,
    cache_dir: str,
    device: torch.device,
) -> tuple[Iterator[torch.Tensor], tuple[int, int, int, int]]:
    llms = build_llms(cfg.activations_harvester.llms, cache_dir, device)

    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_iterator_S = _build_tokens_sequence_iterator(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
    ).get_sequence_iterator()

    # then, shuffle this iterator (only between, not within, sequences) so that we don't have to worry
    # about long documents introducing high feature correlations
    # this shuffler returns batches, hence (B, S)
    shuffled_token_sequence_iterator_BS = batch_shuffle_tensor_iterator_BX(
        tensor_iterator_X=token_sequence_iterator_S,
        shuffle_buffer_size=cfg.sequence_shuffle_buffer_size,
        yield_batch_size=cfg.activations_harvester.harvest_batch_size,
    )

    # then, run these sequences through the model to get activations
    token_activations_iterator_MLD = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=cfg.activations_harvester.layer_indices_to_harvest,
        token_sequence_iterator_BS=shuffled_token_sequence_iterator_BS,
    ).get_token_activations_iterator_MLD()

    # shuffle these token activations, so that we eliminate high feature correlations inside sequences
    shuffled_activations_iterator_BMLD = batch_shuffle_tensor_iterator_BX(
        tensor_iterator_X=token_activations_iterator_MLD,
        shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=cfg.cc_training_batch_size,
    )

    batch_size = cfg.cc_training_batch_size
    num_layers = len(cfg.activations_harvester.layer_indices_to_harvest)
    num_models = len(llms)
    d_model = llms[0].cfg.d_model
    shape = (batch_size, num_layers, num_models, d_model)

    return shuffled_activations_iterator_BMLD, shape


def _build_tokens_sequence_iterator(
    cfg: SequenceIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenSequenceLoader:
    if cfg.classname == "CommonCorpusTokenSequenceIterator":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided for common_corpus")
        if cfg.kwargs["sequence_length"] is None:
            raise ValueError("sequence_length must be provided for common_corpus")
        return CommonCorpusTokenSequenceIterator(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            sequence_length=cfg.kwargs["sequence_length"],
        )
    elif cfg.classname == "ConnorGemma2TokenSequenceLoader":
        return ConnorGemma2TokenSequenceLoader(cache_dir=cache_dir)
    elif cfg.classname == "ToyOverfittingTokenSequenceIterator":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided for common_corpus")
        if cfg.kwargs["sequence_length"] is None:
            raise ValueError("sequence_length must be provided for common_corpus")
        return ToyOverfittingTokenSequenceIterator(sequence_length=cfg.kwargs["sequence_length"])
    raise ValueError(f"Unknown tokens sequence iterator config name: {cfg}")
