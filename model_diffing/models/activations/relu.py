from typing import Any

import torch as t

from model_diffing.models.activations.activation_function import ActivationFunction


class ReLUActivation(ActivationFunction):
    def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
        return t.relu(hidden_preact_BH)

    def _dump_cfg(self) -> dict[str, int | str]:
        return {}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "ReLUActivation":
        return cls()
