from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property

import torch
from einops import rearrange
from transformers import PreTrainedTokenizerBase  # type: ignore  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.data.token_loader import TokenSequenceLoader, build_tokens_sequence_loader
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_jumprelu_sliding_window.config import SlidingWindowDataConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X


class BaseTokenLayerActivationsDataloader(ABC):
    @abstractmethod
    def get_shuffled_activations_iterator_BTLD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def batch_shape_BTLD(self) -> tuple[int, int, int, int]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...

    @abstractmethod
    def get_norm_scaling_factors_TL(self) -> torch.Tensor: ...


class SlidingWindowScaledActivationsDataloader(BaseTokenLayerActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int,
        yield_batch_size: int,
        device: torch.device,
        n_batches_for_norm_estimate: int,
        window_size: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size
        self._window_size = window_size

        # important note: using the raw iterator here, not the scaled one.
        self._norm_scaling_factors_TL = estimate_norm_scaling_factor_X(
            dataloader_BXD=self._shuffled_raw_activations_iterator_BTLD,
            device=device,
            n_batches_for_norm_estimate=n_batches_for_norm_estimate,
        )

        n_models, n_layers, d_model = self._activations_harvester.activation_shape_MLD
        if n_models != 1:
            raise ValueError(
                "ActivationHarvester is configured incorrectly. Should only be harvesting for 1 model for sliding window"
            )
        self.activation_shape_LD = n_layers, d_model

    def get_norm_scaling_factors_TL(self) -> torch.Tensor:
        return self._norm_scaling_factors_TL

    @torch.no_grad()
    def _activations_iterator_TLD(self) -> Iterator[torch.Tensor]:
        for sequences_chunk_BS in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMLD = self._activations_harvester.get_activations_BSMLD(sequences_chunk_BS)
            assert activations_BSMLD.shape[2] == 1, "should only be doing 1 model at a time for sliding window"
            activations_BSLD = activations_BSMLD[:, :, 0]

            B, S, L, D = activations_BSLD.shape

            # sliding window over the sequence dimension, adding a new token dimension
            activations_BSLDT = activations_BSLD.unfold(
                dimension=1,
                size=2,
                step=1,
            )
            activations_BsTLD = rearrange(activations_BSLDT, "b s l d t -> (b s) t l d")
            new_seq_len = S - (self._window_size - 1)
            assert activations_BsTLD.shape == (B * new_seq_len, self._window_size, L, D)
            yield from activations_BsTLD

    @cached_property
    @torch.no_grad()
    def _shuffled_raw_activations_iterator_BTLD(self) -> Iterator[torch.Tensor]:
        activations_iterator_TLD = self._activations_iterator_TLD()

        # shuffle these token activations, so that we eliminate high feature correlations inside sequences
        shuffled_activations_iterator_BTLD = batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=activations_iterator_TLD,
            shuffle_buffer_size=self._activations_shuffle_buffer_size,
            yield_batch_size=self._yield_batch_size,
            name="llm activations",
        )

        return shuffled_activations_iterator_BTLD

    @cached_property
    @torch.no_grad()
    def _shuffled_activations_iterator_BTLD(self) -> Iterator[torch.Tensor]:
        raw_activations_iterator_BTLD = self._shuffled_raw_activations_iterator_BTLD
        scaling_factors_TL1 = rearrange(self.get_norm_scaling_factors_TL(), "t l -> t l 1")
        for unscaled_example_BTLD in raw_activations_iterator_BTLD:
            yield unscaled_example_BTLD * scaling_factors_TL1

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def batch_shape_BTLD(self) -> tuple[int, int, int, int]:
        B = self._yield_batch_size
        T = 2
        L, D = self.activation_shape_LD
        return B, T, L, D

    def get_shuffled_activations_iterator_BTLD(self) -> Iterator[torch.Tensor]:
        return self._shuffled_activations_iterator_BTLD


def build_sliding_window_dataloader(
    cfg: SlidingWindowDataConfig,
    batch_size: int,
    cache_dir: str,
    device: torch.device,
) -> BaseTokenLayerActivationsDataloader:
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
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=cfg.activations_harvester.layer_indices_to_harvest,
    )

    activations_dataloader = SlidingWindowScaledActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        device=device,
        n_batches_for_norm_estimate=cfg.n_batches_for_norm_estimate,
        window_size=cfg.token_window_size,
    )

    return activations_dataloader
