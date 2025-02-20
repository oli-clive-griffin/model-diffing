from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import torch
from einops import rearrange
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.data.token_loader import TokenSequenceLoader, TokensSequenceBatch, build_tokens_sequence_loader
from model_diffing.scripts.config_common import DataConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X


class BaseModelHookpointActivationsDataloader(ABC):
    @abstractmethod
    def get_shuffled_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...

    @abstractmethod
    def get_norm_scaling_factors_MP(self) -> torch.Tensor: ...


class ScaledModelHookpointActivationsDataloader(BaseModelHookpointActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int,
        yield_batch_size: int,
        device: torch.device,
        n_batches_for_norm_estimate: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size

        # important note: using the raw iterator here, not the scaled one.
        self._norm_scaling_factors_MP = estimate_norm_scaling_factor_X(
            self._shuffled_raw_activations_iterator_BMPD,
            device,
            n_batches_for_norm_estimate,
        )

    def get_norm_scaling_factors_MP(self) -> torch.Tensor:
        return self._norm_scaling_factors_MP

    @dataclass
    class ActivationsWithText:
        activations_BSMPD: torch.Tensor
        tokens_BS: torch.Tensor

    @torch.no_grad()
    def iterate_activations_with_text(self) -> Iterator[ActivationsWithText]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMPD = self._activations_harvester.get_activations_BSMPD(seq.tokens_BS)
            yield self.ActivationsWithText(activations_BSMPD=activations_BSMPD, tokens_BS=seq.tokens_BS)

    @torch.no_grad()
    def _activations_iterator_MPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMPD = self._activations_harvester.get_activations_BSMPD(seq.tokens_BS)

            activations_BsMPD = rearrange(activations_BSMPD, "b s m p d -> (b s) m p d")
            special_tokens_mask_Bs = rearrange(seq.special_tokens_mask_BS, "b s -> (b s)")
            activations_BsMPD = activations_BsMPD[~special_tokens_mask_Bs]

            yield from activations_BsMPD

    @cached_property
    @torch.no_grad()
    def _shuffled_raw_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]:
        activations_iterator_MPD = self._activations_iterator_MPD()

        # shuffle these token activations, so that we eliminate high feature correlations inside sequences
        shuffled_activations_iterator_BMPD = batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=activations_iterator_MPD,
            shuffle_buffer_size=self._activations_shuffle_buffer_size,
            yield_batch_size=self._yield_batch_size,
            name="llm activations",
        )

        return shuffled_activations_iterator_BMPD

    @cached_property
    @torch.no_grad()
    def _shuffled_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]:
        raw_activations_iterator_BMPD = self._shuffled_raw_activations_iterator_BMPD
        scaling_factors_MP1 = rearrange(self.get_norm_scaling_factors_MP(), "m p -> m p 1")
        for unscaled_example_BMPD in raw_activations_iterator_BMPD:
            yield unscaled_example_BMPD * scaling_factors_MP1

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def get_shuffled_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]:
        return self._shuffled_activations_iterator_BMPD


def build_dataloader(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    hookpoints: list[str],
    batch_size: int,
    cache_dir: str,
    device: torch.device,
) -> ScaledModelHookpointActivationsDataloader:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
    )

    activations_dataloader = ScaledModelHookpointActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        device=device,
        n_batches_for_norm_estimate=cfg.n_batches_for_norm_estimate,
    )

    return activations_dataloader
