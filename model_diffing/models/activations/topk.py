from typing import Any

import torch as t
from einops import rearrange

from model_diffing.log import logger
from model_diffing.models.activations.activation_function import ActivationFunction


class TopkActivation(ActivationFunction):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
        _topk_values_BH, topk_indices_BH = hidden_preact_BH.topk(self.k, dim=-1, sorted=False)
        hidden_BH = t.zeros_like(hidden_preact_BH)
        hidden_BH.scatter_(-1, topk_indices_BH, _topk_values_BH)
        # TODO: use faster implementation as in https://github.com/EleutherAI/sparsify
        return hidden_BH

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"k": self.k}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "TopkActivation":
        return cls(cfg["k"])


class BatchTopkActivation(ActivationFunction):
    def __init__(self, k_per_example: int):
        super().__init__()
        self.k_per_example = k_per_example

    def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
        batch_size = hidden_preact_BH.shape[0]
        batch_k = self.k_per_example * batch_size
        hidden_preact_Bh = rearrange(hidden_preact_BH, "batch hidden -> (batch hidden)")
        topk_values_Bh, topk_indices_Bh = hidden_preact_Bh.topk(k=batch_k, sorted=False)
        hidden_Bh = t.zeros_like(hidden_preact_Bh)
        hidden_Bh.scatter_(-1, topk_indices_Bh, topk_values_Bh)
        hidden_BH = rearrange(hidden_Bh, "(batch hidden) -> batch hidden", batch=batch_size)
        return hidden_BH

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"k_per_example": self.k_per_example}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "BatchTopkActivation":
        return cls(cfg["k_per_example"])


class GroupMaxActivation(ActivationFunction):
    def __init__(self, k_groups: int, hidden_size: int):
        super().__init__()
        self.k_groups = k_groups
        self.hidden_size = hidden_size
        self.offsets_K = t.arange(0, self.hidden_size, self.hidden_size // self.k_groups)
        logger.warn("using topk activation, BatchTopk is available and generally ≈better")

    def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
        # values, indices = hidden_preact_BH.unflatten(-1, (self.cfg.k, -1)).max(dim=-1)
        hidden_preact_BKHg = rearrange(
            hidden_preact_BH,
            "b (k_groups h_per_group) -> b k_groups h_per_group",
            k_groups=self.k_groups,
            h_per_group=self.hidden_size // self.k_groups,
        )

        group_max_values_BK, group_indices_BK = hidden_preact_BKHg.max(dim=-1)

        # torch.max gives us indices into each group, but we want indices into the
        # flattened tensor. Add the offsets to get the correct indices.
        indices_BK = group_indices_BK + self.offsets_K

        hidden_BH = t.zeros_like(hidden_preact_BH)
        hidden_BH.scatter_(-1, indices_BK, group_max_values_BK)
        return hidden_BH

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"k_groups": self.k_groups, "hidden_size": self.hidden_size}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "GroupMaxActivation":
        return cls(cfg["k_groups"], cfg["hidden_size"])


# ideas:
# probably not going to work super well like this. Need to do all / each k reconstructions at once

# class MatryoshkaTopKActivation(ActivationFunction):
#     def __init__(self, get_k: Callable[[], int] | list[int]):
#         super().__init__()
#         self.get_k = get_k if callable(get_k) else lambda: random.choice(get_k)

#     def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
#         _topk_values_BH, topk_indices_BH = hidden_preact_BH.topk(self.get_k(), dim=-1, sorted=False)
#         hidden_BH = t.zeros_like(hidden_preact_BH)
#         hidden_BH.scatter_(-1, topk_indices_BH, _topk_values_BH)
#         # TODO: use faster implementation as in https://github.com/EleutherAI/sparsify
#         return hidden_BH


# class MatryoshkaBatchTopKActivation(ActivationFunction):
#     def __init__(self, get_k: Callable[[], int] | list[int]):
#         super().__init__()
#         self.get_k = get_k if callable(get_k) else lambda: random.choice(get_k)

#     def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
#         batch_size = hidden_preact_BH.shape[0]
#         batch_k = self.get_k() * batch_size
#         hidden_preact_Bh = rearrange(hidden_preact_BH, "batch hidden -> (batch hidden)")
#         topk_values_Bh, topk_indices_Bh = hidden_preact_Bh.topk(k=batch_k, sorted=False)
#         hidden_Bh = t.zeros_like(hidden_preact_Bh)
#         hidden_Bh.scatter_(-1, topk_indices_Bh, topk_values_Bh)
#         return rearrange(hidden_Bh, "(batch hidden) -> batch hidden", batch=batch_size)
