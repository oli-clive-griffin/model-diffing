from typing import Literal

import torch
from transformer_lens import HookedTransformer  # type: ignore

from model_diffing.data.activation_cache import ActivationsCache
from model_diffing.log import logger

# shapes:
# B: batch size
# S: sequence length
# P: hookpoints
# D: model d_model

CacheMode = Literal["no_cache", "cache", "cache_with_mmap"]


class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer],
        hookpoints: list[str],
        cache_dir: str | None = None,
        cache_mode: CacheMode = "no_cache",
    ):
        if len({llm.cfg.d_model for llm in llms}) != 1:
            raise ValueError("All models must have the same d_model")
        self._llms = llms
        self._hookpoints = hookpoints

        # Set up the activations cache
        self._cache = None
        if cache_mode != "no_cache":
            if not cache_dir:
                raise ValueError("cache_mode is enabled but no cache_dir provided; caching will be disabled")
            self._cache = ActivationsCache(cache_dir=cache_dir, use_mmap=cache_mode == "cache_with_mmap")

        self.num_models = len(llms)
        self._num_hookpoints = len(hookpoints)
        self._layer_to_stop_at = self._get_layer_to_stop_at()

    def _get_layer_to_stop_at(self) -> int:
        last_needed_layer = max(_get_layer(name) for name in self._hookpoints)
        layer_to_stop_at = last_needed_layer + 1
        logger.info(f"computed last needed layer: {last_needed_layer}, stopping at {layer_to_stop_at}")
        return layer_to_stop_at

    def _names_filter(self, name: str) -> bool:
        return name in self._hookpoints  # not doing any fancy hash/set usage as this list is tiny

    def compute_model_activations(
        self,
        model: HookedTransformer,
        sequence_BS: torch.Tensor,
        attention_mask_BS: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute activations by running the model. No caching involved."""
        _, cache = model.run_with_cache(
            sequence_BS,
            names_filter=self._names_filter,
            stop_at_layer=self._layer_to_stop_at,
            attention_mask=attention_mask_BS,
        )
        # cache[name] is shape BSD, so stacking on dim 2 = BSPD
        activations_BSPD = torch.stack([cache[name] for name in self._hookpoints], dim=2)  # adds hookpoint dim (P)
        return activations_BSPD

    def _get_model_activations_BSPD(
        self,
        model: HookedTransformer,
        sequence_BS: torch.Tensor,
        attention_mask_BS: torch.Tensor | None,
    ) -> torch.Tensor:
        # Check if we can load from cache
        if self._cache:
            cache_key = self._cache.get_cache_key(model, sequence_BS)
            activations_BSPD = self._cache.load_activations(cache_key, sequence_BS.device)

            if activations_BSPD is None:
                activations_BSPD = self.compute_model_activations(model, sequence_BS, attention_mask_BS)
                self._cache.save_activations(cache_key, activations_BSPD)

            return activations_BSPD

        # Compute activations if not cached or cache loading failed
        activations_BSPD = self.compute_model_activations(model, sequence_BS, attention_mask_BS)

        return activations_BSPD

    def get_activations_BSMPD(self, sequence_BS: torch.Tensor, attention_mask_BS: torch.Tensor | None) -> torch.Tensor:
        activations = [self._get_model_activations_BSPD(model, sequence_BS, attention_mask_BS) for model in self._llms]
        activations_BSMPD = torch.stack(activations, dim=2)
        return activations_BSMPD


def _get_layer(hookpoint: str) -> int:
    if "blocks" not in hookpoint:
        raise NotImplementedError(
            f'Hookpoint "{hookpoint}" is not a blocks hookpoint, cannot determine layer, (but feel free to add this functionality!)'
        )
    assert hookpoint.startswith("blocks.")
    return int(hookpoint.split(".")[1])
