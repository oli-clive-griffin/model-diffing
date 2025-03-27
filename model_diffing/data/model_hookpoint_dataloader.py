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
from model_diffing.utils import change_batch_size_BX


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
        yield_batch_size_B: int,
        n_tokens_for_norm_estimate: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._yield_batch_size_B = yield_batch_size_B
        self._device = self._activations_harvester._llms[0].W_E.device

        norm_scaling_factors_MP = estimate_norm_scaling_factor_X(
            self._activations_iterator_BMPD(),  # don't pass the scaling factors here (because we're computing them!)
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

    def _activations_iterator_HsMPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_HSMPD = self._activations_harvester.get_activations_HSMPD(seq.tokens_HS)
            activations_HsMPD = rearrange(activations_HSMPD, "h s m p d -> (h s) m p d")
            special_tokens_mask_Hs = rearrange(seq.special_tokens_mask_HS, "h s -> (h s)")
            yield activations_HsMPD[~special_tokens_mask_Hs]

    def _activations_iterator_BMPD(self, scaling_factors_MP: torch.Tensor | None = None) -> Iterator[torch.Tensor]:
        iterator_HsMPD = self._activations_iterator_HsMPD()
        batch_iter_BMPD = change_batch_size_BX(iterator_HX=iterator_HsMPD, new_batch_size_B=self._yield_batch_size_B)

        if scaling_factors_MP is None:
            scaling_factors_MP1 = torch.ones((1, 1, 1), device=self._device)
        else:
            scaling_factors_MP1 = rearrange(scaling_factors_MP.to(self._device), "m p -> m p 1")

        cuda_avail = torch.cuda.is_available()
        for i, batch_BMPD in enumerate(batch_iter_BMPD):
            yield batch_BMPD * scaling_factors_MP1
            if i % 5 == 0 and cuda_avail:
                torch.cuda.empty_cache()


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

    activations_dataloader = ScaledModelHookpointActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        yield_batch_size_B=batch_size,
        n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
    )

    return activations_dataloader
