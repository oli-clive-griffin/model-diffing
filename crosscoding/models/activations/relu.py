from typing import Any

import torch

from crosscoding.models.activations.activation_function import ActivationFunction


class ReLUActivation(ActivationFunction):
    def forward(self, latent_preact_BL: torch.Tensor) -> torch.Tensor:
        return torch.relu(latent_preact_BL)

    def _dump_cfg(self) -> dict[str, int | str]:
        return {}

    @classmethod
    def _scaffold_from_cfg(cls, cfg: dict[str, Any]) -> "ReLUActivation":
        return cls()
