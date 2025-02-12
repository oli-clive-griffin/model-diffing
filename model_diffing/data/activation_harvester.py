from functools import cached_property

import torch
from transformer_lens import HookedTransformer  # type: ignore

# shapes:
# B: batch size
# S: sequence length
# P: hookpoints
# D: model d_model


class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer],
        hookpoints: list[str],
    ):
        if len({llm.cfg.d_model for llm in llms}) != 1:
            raise ValueError("All models must have the same d_model")
        self._llms = llms
        self._hookpoints = hookpoints

        self.num_models = len(llms)
        self._num_hookpoints = len(hookpoints)

    @cached_property
    def names_set(self) -> set[str]:
        return set(self._hookpoints)

    def _names_filter(self, name: str) -> bool:
        return name in self.names_set

    def _get_model_activations_BSPD(self, model: HookedTransformer, sequence_BS: torch.Tensor) -> torch.Tensor:
        _, cache = model.run_with_cache(sequence_BS, names_filter=self._names_filter)
        # cache[name] is shape BSD, so stacking on dim 2 = BSPD
        activations_BSPD = torch.stack([cache[name] for name in self._hookpoints], dim=2)  # adds hookpoint dim (P)
        return activations_BSPD

    def get_activations_BSMPD(self, sequence_BS: torch.Tensor) -> torch.Tensor:
        activations = [self._get_model_activations_BSPD(model, sequence_BS) for model in self._llms]
        activations_BSMPD = torch.stack(activations, dim=2)
        return activations_BSMPD
