from typing import Any

import torch as t

from model_diffing.utils import SaveableModule


class ReLUActivation(SaveableModule):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x)

    def _dump_cfg(self) -> dict[str, int | str]:
        return {}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "ReLUActivation":
        return cls()
