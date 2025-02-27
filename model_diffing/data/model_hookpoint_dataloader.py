import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
from einops import rearrange
from transformer_lens import HookedTransformer  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.token_loader import TokenSequenceLoader, build_tokens_sequence_loader
from model_diffing.log import logger
from model_diffing.scripts.config_common import DataConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X
from model_diffing.utils import iter_into_batches


class BaseModelHookpointActivationsDataloader(ABC):
    @abstractmethod
    def get_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...

    @abstractmethod
    def get_norm_scaling_factors_MP(self) -> torch.Tensor: ...


class ScaledModelHookpointActivationsDataloader(BaseModelHookpointActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int | None,
        yield_batch_size: int,
        n_tokens_for_norm_estimate: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size

        norm_scaling_factors_MP = estimate_norm_scaling_factor_X(
            self._activations_iterator_BMPD(),  # don't pass the scaling factors here (becuase we're computing them!)
            n_tokens_for_norm_estimate,
        )

        self._norm_scaling_factors_MP = norm_scaling_factors_MP
        self._iterator = self._activations_iterator_BMPD(norm_scaling_factors_MP)

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def get_activations_iterator_BMPD(self) -> Iterator[torch.Tensor]:
        return self._iterator

    def get_norm_scaling_factors_MP(self) -> torch.Tensor:
        return self._norm_scaling_factors_MP

    @torch.no_grad()
    def _activations_iterator_BsMPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMPD = self._activations_harvester.get_activations_BSMPD(seq.tokens_BS, seq.attention_mask_BS)
            activations_BsMPD = rearrange(activations_BSMPD, "b s m p d -> (b s) m p d")
            special_tokens_mask_Bs = rearrange(seq.special_tokens_mask_BS, "b s -> (b s)")
            yield activations_BsMPD[~special_tokens_mask_Bs]

    @torch.no_grad()
    def _activations_iterator_BMPD(self, scaling_factors_MP: torch.Tensor | None = None) -> Iterator[torch.Tensor]:
        iterator_BsMPD = self._activations_iterator_BsMPD()

        device = next(iterator_BsMPD).device

        if scaling_factors_MP is None:
            scaling_factors_MP1 = torch.ones((1, 1, 1), device=device)
        else:
            scaling_factors_MP1 = rearrange(scaling_factors_MP, "m p -> m p 1").to(device)
        
        for batch_BMPD in iter_into_batches(iterator_BsMPD, self._yield_batch_size):
            yield batch_BMPD * scaling_factors_MP1


def build_dataloader(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    hookpoints: list[str],
    batch_size: int,
    cache_dir: str,
) -> ScaledModelHookpointActivationsDataloader:
    tokenizer = llms[0].tokenizer
    assert all(
        llm.tokenizer.special_tokens_map == tokenizer.special_tokens_map  # type: ignore
        for llm in llms
    ), "all tokenizers should have the same special tokens"
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
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

    activations_dataloader = ScaledModelHookpointActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
    )

    return activations_dataloader
