from typing import Literal, overload

import torch
from transformer_lens import HookedTransformer  # type: ignore

from model_diffing.data.activation_cache import ActivationsCache
from model_diffing.log import logger

# shapes:
# H: (harvesting) batch size
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
        self._device = llms[0].W_E.device
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

    def get_activations_HSMPD(
        self,
        sequence_HS: torch.Tensor,
    ) -> torch.Tensor:
        MPD = (len(self._llms), len(self._hookpoints), self._llms[0].cfg.d_model)
        activations_HSMPD = torch.empty(*sequence_HS.shape, *MPD, device=self._device)
        for m, model in enumerate(self._llms):
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    sequence_HS.to(self._device),
                    names_filter=self._names_filter,
                    stop_at_layer=self._layer_to_stop_at,
                )
            for p, hookpoint in enumerate(self._hookpoints):
                hookpoint_act_HSD = cache[hookpoint]
                activations_HSMPD[:, :, m, p, :] = hookpoint_act_HSD
        return activations_HSMPD


def _get_layer(hookpoint: str) -> int:
    if "blocks" not in hookpoint:
        raise NotImplementedError(
            f'Hookpoint "{hookpoint}" is not a blocks hookpoint, cannot determine layer, (but feel free to add this functionality!)'
        )
    assert hookpoint.startswith("blocks.")
    return int(hookpoint.split(".")[1])
