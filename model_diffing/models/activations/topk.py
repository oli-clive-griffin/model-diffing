from typing import Any

import torch as t
from einops import rearrange

from model_diffing.utils import SaveableModule


class TopkActivation(SaveableModule):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, hidden_preactivation_BH: t.Tensor) -> t.Tensor:
        _topk_values_BH, topk_indices_BH = hidden_preactivation_BH.topk(self.k, dim=-1)
        hidden_BH = t.zeros_like(hidden_preactivation_BH)
        hidden_BH.scatter_(-1, topk_indices_BH, _topk_values_BH)
        # TODO: use faster implementation as in https://github.com/EleutherAI/sparsify
        return hidden_BH

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"k": self.k}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "TopkActivation":
        return cls(cfg["k"])


class BatchTopkActivation(SaveableModule):
    def __init__(self, k_per_example: int):
        super().__init__()
        self.k_per_example = k_per_example

    def forward(self, hidden_preactivation_BH: t.Tensor) -> t.Tensor:
        batch_size = hidden_preactivation_BH.shape[0]
        batch_k = self.k_per_example * batch_size
        hidden_preactivation_Bh = rearrange(hidden_preactivation_BH, "batch hidden -> (batch hidden)")
        _topk_values_Bh, topk_indices_Bh = hidden_preactivation_Bh.topk(k=batch_k)
        hidden_Bh = t.zeros_like(hidden_preactivation_Bh)
        hidden_Bh.scatter_(-1, topk_indices_Bh, _topk_values_Bh)
        hidden_BH = rearrange(hidden_Bh, "(batch hidden) -> batch hidden", batch=batch_size)
        return hidden_BH

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"k_per_example": self.k_per_example}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "BatchTopkActivation":
        return cls(cfg["k_per_example"])
