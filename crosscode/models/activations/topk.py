from typing import Any, Literal, Protocol

import torch
from einops import rearrange

from crosscode.log import logger
from crosscode.models.activations.activation_function import ActivationFunction


class TopkStyleActivation(Protocol):
    def __call__(self, x_BL: torch.Tensor, k: int) -> torch.Tensor: ...


def topk_activation(x_BL: torch.Tensor, k: int) -> torch.Tensor:
    values, indices = x_BL.topk(k, dim=-1, sorted=False)
    out_BL = torch.zeros_like(x_BL)
    out_BL.scatter_(-1, indices, values)
    return out_BL


def groupmax_activation(x_BL: torch.Tensor, n_groups: int) -> torch.Tensor:
    latents_size = x_BL.shape[1]
    assert latents_size % n_groups == 0, "latents_size must be divisible by n_groups"
    latents_per_group = latents_size // n_groups

    x_BKLg = rearrange(
        x_BL,
        "b (n_groups latents_per_group) -> b n_groups latents_per_group",
        n_groups=n_groups,
        latents_per_group=latents_per_group,
    )

    values_BK, indices_BK = x_BKLg.max(dim=-1)

    # torch.max gives us indices into each group, but we want indices into the
    # flattened tensor. Add the offsets to get the correct indices.
    offsets_K = torch.arange(0, latents_size, latents_per_group)

    indices_BK = indices_BK + offsets_K

    out_BL = torch.zeros_like(x_BL)
    out_BL.scatter_(-1, indices_BK, values_BK)
    return out_BL


def batch_topk_activation(x_BL: torch.Tensor, k: int) -> torch.Tensor:
    batch_size = x_BL.shape[0]
    batch_k = k * batch_size
    latent_preact_Bl = rearrange(x_BL, "batch latent -> (batch latent)")
    values_Bh, indices_Bh = latent_preact_Bl.topk(k=batch_k, sorted=False)
    out_Bh = torch.zeros_like(latent_preact_Bl)
    out_Bh.scatter_(-1, indices_Bh, values_Bh)
    out_BL = rearrange(out_Bh, "(batch latent) -> batch latent", batch=batch_size)
    return out_BL


class TopkActivation(ActivationFunction, TopkStyleActivation):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, latent_preact_BL: torch.Tensor) -> torch.Tensor:
        return topk_activation(latent_preact_BL, self.k)

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"k": self.k}

    @classmethod
    def _scaffold_from_cfg(cls, cfg: dict[str, Any]) -> "TopkActivation":
        return cls(cfg["k"])


class BatchTopkActivation(ActivationFunction):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, latent_preact_BL: torch.Tensor) -> torch.Tensor:
        return batch_topk_activation(latent_preact_BL, self.k)

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"k": self.k}

    @classmethod
    def _scaffold_from_cfg(cls, cfg: dict[str, Any]) -> "BatchTopkActivation":
        return cls(cfg["k"])


class GroupMaxActivation(ActivationFunction):
    def __init__(self, n_groups: int):
        super().__init__()
        self.n_groups = n_groups
        logger.warn("using topk activation, BatchTopk is available and generally ≈better")

    def forward(self, latent_preact_BL: torch.Tensor) -> torch.Tensor:
        return groupmax_activation(latent_preact_BL, self.n_groups)

    def _dump_cfg(self) -> dict[str, int | str]:
        return {"n_groups": self.n_groups}

    @classmethod
    def _scaffold_from_cfg(cls, cfg: dict[str, Any]) -> "GroupMaxActivation":
        return cls(cfg["n_groups"])


TopKStyle = Literal["topk", "batch_topk", "groupmax"]
