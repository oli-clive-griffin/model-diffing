import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
from einops import rearrange
from transformer_lens import HookedTransformer  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.token_loader import TokenSequenceLoader, build_tokens_sequence_loader
from model_diffing.log import logger
from model_diffing.scripts.config_common import DataConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X
from model_diffing.utils import change_batch_size_BX


class BaseTokenHookpointActivationsDataloader(ABC):
    @abstractmethod
    def get_activations_iterator_BTPD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...

    @abstractmethod
    def get_norm_scaling_factors_TP(self) -> torch.Tensor: ...


class SlidingWindowScaledActivationsDataloader(BaseTokenHookpointActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        yield_batch_size: int,
        n_tokens_for_norm_estimate: int,
        window_size: int,
    ):
        if activations_harvester.num_models != 1:
            raise ValueError(
                "ActivationHarvester is configured incorrectly. "
                "Should only be harvesting for 1 model for sliding window"
            )

        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._yield_batch_size = yield_batch_size
        self._window_size_tokens = window_size

        norm_scaling_factors_TP = estimate_norm_scaling_factor_X(
            dataloader_BXD=self._activations_iterator_BTPD(),  # don't pass the scaling factors here (because we're computing them!)
            n_tokens_for_norm_estimate=n_tokens_for_norm_estimate,
        )
        self._norm_scaling_factors_TP = norm_scaling_factors_TP
        self._iterator = self._activations_iterator_BTPD(self._norm_scaling_factors_TP)

    def get_activations_iterator_BTPD(self) -> Iterator[torch.Tensor]:
        return self._activations_iterator_BTPD()

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def get_norm_scaling_factors_TP(self) -> torch.Tensor:
        return self._norm_scaling_factors_TP

    def _activations_iterator_HsTPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_HSMPD = self._activations_harvester.get_activations_HSMPD(seq.tokens_HS)

            # pick only the first (and only) model's activations
            assert activations_HSMPD.shape[2] == 1, "should only be doing 1 model at a time for sliding window"
            activations_HSPD = activations_HSMPD[:, :, 0]

            # flatten across batch and sequence dimensions, and filter out special tokens
            activations_HsPD = rearrange(activations_HSPD, "h s p d -> (h s) p d")
            special_tokens_mask_Hs = rearrange(seq.special_tokens_mask_HS, "h s -> (h s)")
            activations_HsPD = activations_HsPD[~special_tokens_mask_Hs]

            # sliding window over the sequence dimension, adding a new token dimension
            # new_seq_len = S - (self._window_size_tokens - 1)
            activations_HsPDT = activations_HsPD.unfold(dimension=0, size=2, step=1)
            activations_HsTPD = rearrange(activations_HsPDT, "hs p d t -> hs t p d")

            yield activations_HsTPD

    def _activations_iterator_BTPD(self, scaling_factors_TP: torch.Tensor | None = None) -> Iterator[torch.Tensor]:
        iterator_HsTPD = self._activations_iterator_HsTPD()

        device = next(iterator_HsTPD).device

        if scaling_factors_TP is None:
            scaling_factors_TP1 = torch.ones((1, 1, 1), device=device)
        else:
            scaling_factors_TP1 = rearrange(scaling_factors_TP, "t p -> t p 1").to(device)

        for batch_BTPD in change_batch_size_BX(iterator_HX=iterator_HsTPD, new_batch_size_B=self._yield_batch_size):
            assert batch_BTPD.shape[0] == self._yield_batch_size, (
                f"batch_BTPD.shape[0] {batch_BTPD.shape[0]} != self._yield_batch_size {self._yield_batch_size}"
            )  # REMOVE ME
            yield batch_BTPD * scaling_factors_TP1


def build_sliding_window_dataloader(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    hookpoints: list[str],
    batch_size: int,
    cache_dir: str,
    window_size: int,
) -> BaseTokenHookpointActivationsDataloader:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.token_sequence_loader,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # Create activations cache directory if cache is enabled
    activations_cache_dir = None
    if cfg.activations_harvester.cache_mode != "no_cache":
        activations_cache_dir = os.path.join(cache_dir, "activations_cache")
        logger.info(f"Activations will be cached in: {activations_cache_dir}")

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
        cache_dir=activations_cache_dir,
        cache_mode=cfg.activations_harvester.cache_mode,
    )

    activations_dataloader = SlidingWindowScaledActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        yield_batch_size=batch_size,
        n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
        window_size=window_size,
    )

    return activations_dataloader
