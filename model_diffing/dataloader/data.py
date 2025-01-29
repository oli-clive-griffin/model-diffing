import torch
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.dataloader.activations import ActivationsDataloader, ActivationsHarvester
from model_diffing.dataloader.token_loader import (
    CommonCorpusTokenSequenceLoader,
    ConnorGemma2TokenSequenceLoader,
    TokenSequenceLoader,
    ToyOverfittingTokenSequenceLoader,
)
from model_diffing.scripts.config_common import DataConfig, SequenceIteratorConfig
from model_diffing.scripts.llms import build_llms


def build_dataloader(
    cfg: DataConfig,
    cache_dir: str,
    device: torch.device,
) -> ActivationsDataloader:
    llms = build_llms(
        cfg.activations_harvester.llms,
        cache_dir,
        device,
        dtype=cfg.activations_harvester.inference_dtype,
    )

    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = _build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
    )

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=cfg.activations_harvester.layer_indices_to_harvest,
    )

    activations_dataloader = ActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=cfg.cc_training_batch_size,
    )

    return activations_dataloader


def _build_tokens_sequence_loader(
    cfg: SequenceIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenSequenceLoader:
    if cfg.classname == "CommonCorpusTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided for common_corpus")
        if cfg.kwargs["sequence_length"] is None:
            raise ValueError("sequence_length must be provided for common_corpus")
        if cfg.kwargs["shuffle_buffer_size"] is None:
            raise ValueError("shuffle_buffer_size must be provided for common_corpus")
        return CommonCorpusTokenSequenceLoader(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
            **cfg.kwargs,
        )
    elif cfg.classname == "ConnorGemma2TokenSequenceLoader":
        return ConnorGemma2TokenSequenceLoader(
            cache_dir=cache_dir,
            batch_size=cfg.batch_size,
        )
    elif cfg.classname == "ToyOverfittingTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided for common_corpus")
        if cfg.kwargs["sequence_length"] is None:
            raise ValueError("sequence_length must be provided for common_corpus")
        return ToyOverfittingTokenSequenceLoader(
            batch_size=cfg.batch_size,
            **cfg.kwargs,
        )
    raise ValueError(f"Unknown tokens sequence iterator config name: {cfg}")
