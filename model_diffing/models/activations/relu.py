from typing import Any

import torch as t

from model_diffing.models.activations.activation_function import ActivationFunction


class ReLUActivation(ActivationFunction):
    def forward(self, latent_preact_BL: t.Tensor) -> t.Tensor:
        return t.relu(latent_preact_BL)

    def _dump_cfg(self) -> dict[str, int | str]:
        return {}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "ReLUActivation":
        return cls()
