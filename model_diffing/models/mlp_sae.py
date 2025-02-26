from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from typing import Any, Generic, TypeVar, cast

import torch as t
from einops import einsum, reduce
from torch import nn

from model_diffing.models.activations import ACTIVATIONS_MAP
from model_diffing.utils import SaveableModule, l2_norm

"""
Dimensions:
- B: batch size
- : arbitrary number of crosscoding dimensions
- H: Autoencoder h dimension
- D: Autoencoder d dimension
"""

TActivation = TypeVar("TActivation", bound=SaveableModule)


class MLPSAEcausalCrosscoder(SaveableModule, Generic[TActivation]):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        hidden_activation: TActivation,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

        self.W_dec_HD = nn.Parameter(t.empty((hidden_dim, d_model)))
        self.b_enc_H = nn.Parameter(t.empty((hidden_dim,)))
        self.W_enc_DH = nn.Parameter(t.empty((d_model, hidden_dim)))
        self.b_dec_D = nn.Parameter(t.empty((d_model)))

    def _encode_BH(self, activation_BD: t.Tensor) -> t.Tensor:
        pre_bias_BH = einsum(activation_BD, self.W_enc_DH, "b d, d h -> b h")
        pre_activation_BH = pre_bias_BH + self.b_enc_H
        return self.hidden_activation(pre_activation_BH)

    def _decode_BD(self, hidden_BH: t.Tensor) -> t.Tensor:
        pre_bias_BD = einsum(hidden_BH, self.W_dec_HD, "b h, h d -> b d")
        return pre_bias_BD + self.b_dec_D

    @dataclass
    class ForwardResult:
        hidden_BH: t.Tensor
        output_BD: t.Tensor

    def _forward(self, activation_BD: t.Tensor) -> ForwardResult:
        hidden_BH = self._encode_BH(activation_BD)
        output_BD = self._decode_BD(hidden_BH)

        return self.ForwardResult(
            hidden_BH=hidden_BH,
            output_BD=output_BD,
        )

    def forward_train(
        self,
        activation_BD: t.Tensor,
    ) -> ForwardResult:
        """returns the activations, the h states, and the reconstructed activations"""
        res = self._forward(activation_BD)

        if res.output_BD.shape != activation_BD.shape:
            raise ValueError(f"output_BD.shape {res.output_BD.shape} != activation_BD.shape {activation_BD.shape}")

        return res

    def forward(self, activation_BD: t.Tensor) -> t.Tensor:
        return self._forward(activation_BD).output_BD