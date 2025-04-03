from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import torch
from einops import einsum
from torch import nn

from crosscode.models.activations import ActivationFunction
from crosscode.models.initialization.init_strategy import InitStrategy
from crosscode.saveable_module import SaveableModule

"""
Dimensions:
- B: batch size
- Xi: arbitrary number of input crosscoding dimensions
- Di: input vector space dimensionality
- L: Latents, AKA "features"
- Xo: arbitrary number of output crosscoding dimensions
- Do: output vector space dimensionality
"""


TActivation = TypeVar("TActivation", bound=ActivationFunction)


class BaseCrosscoder(Generic[TActivation], SaveableModule):
    is_folded: torch.Tensor
    folded_scaling_factors_in_Xi: torch.Tensor | None
    folded_scaling_factors_out_Xo: torch.Tensor | None

    def __init__(
        self,
        in_crosscoding_dims: tuple[int, ...],
        d_in: int,
        out_crosscoding_dims: tuple[int, ...],
        d_out: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool = True,
        use_decoder_bias: bool = True,
        init_strategy: InitStrategy["BaseCrosscoder[TActivation]"] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._in_crosscoding_dims = in_crosscoding_dims
        self._out_crosscoding_dims = out_crosscoding_dims
        self._n_latents = n_latents
        self._activation_fn = activation_fn
        self._dtype = dtype

        self._W_enc_XiDiL = nn.Parameter(torch.empty((*in_crosscoding_dims, d_in, n_latents), dtype=dtype))
        self._W_dec_LXoDo = nn.Parameter(torch.empty((n_latents, *out_crosscoding_dims, d_out), dtype=dtype))

        # public because no implementations rename it
        self.b_enc_L = nn.Parameter(torch.empty((n_latents,), dtype=dtype)) if use_encoder_bias else None
        self._b_dec_XoDo = (
            nn.Parameter(torch.empty((*out_crosscoding_dims, d_out), dtype=dtype)) if use_decoder_bias else None
        )

        if init_strategy is not None:
            init_strategy.init_weights(self)

        # Initialize buffers with a tensors of the correct shapes, this means it's always serialized
        self.register_buffer("folded_scaling_factors_in_Xi", torch.zeros(self._in_crosscoding_dims, dtype=dtype))
        self.register_buffer("folded_scaling_factors_out_Xo", torch.zeros(self._out_crosscoding_dims, dtype=dtype))
        # We also track whether it's actually holding a meaningful value by using this boolean flag.
        # Represented as a tensor so that it's serialized by torch.save
        self.register_buffer("is_folded", torch.tensor(False, dtype=torch.bool))

    @property
    def n_latents(self) -> int:
        return self._n_latents

    @property
    def activation_fn(self) -> TActivation:
        return self._activation_fn

    @property
    def device(self) -> torch.device:
        return self._W_dec_LXoDo.device

    @dataclass
    class _ForwardResult:
        pre_activations_BL: torch.Tensor
        latents_BL: torch.Tensor
        output_BXoDo: torch.Tensor

    def get_pre_bias_BL(self, activation_BXiDi: torch.Tensor) -> torch.Tensor:
        return einsum(activation_BXiDi, self._W_enc_XiDiL, "b ..., ... l -> b l")

    def get_preacts_BL(self, activation_BXiDi: torch.Tensor) -> torch.Tensor:
        pre_activations_BL = self.get_pre_bias_BL(activation_BXiDi)

        if self.b_enc_L is not None:
            pre_activations_BL += self.b_enc_L
        return pre_activations_BL


    def decode_BXoDo(self, latents_BL: torch.Tensor) -> torch.Tensor:
        pre_bias_BXoDo = einsum(latents_BL, self._W_dec_LXoDo, "b l, l ... -> b ...")
        if self._b_dec_XoDo is not None:
            pre_bias_BXoDo += self._b_dec_XoDo
        return pre_bias_BXoDo

    def _forward_train(
        self,
        activation_BXiDi: torch.Tensor,
    ) -> _ForwardResult:
        """returns the activations, the latents, and the reconstructed activations"""
        assert activation_BXiDi.shape[1:-1] == self._in_crosscoding_dims

        pre_activations_BL = self.get_preacts_BL(activation_BXiDi)

        latents_BL = self.activation_fn.forward(pre_activations_BL)

        output_BXoDo = self.decode_BXoDo(latents_BL)

        assert output_BXoDo.shape[1:-1] == self._out_crosscoding_dims

        return self._ForwardResult(
            pre_activations_BL=pre_activations_BL,
            latents_BL=latents_BL,
            output_BXoDo=output_BXoDo,
        )

    # def forward(self, activation_BXiDi: torch.Tensor) -> torch.Tensor:
    #     return self.forward_train(activation_BXiDi).output_BXoDo

    @torch.no_grad()
    def _fold_activation_scaling_into_weights_(
        self, scaling_factors_in_Xi: torch.Tensor, scaling_factors_out_Xo: torch.Tensor
    ) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the m can be run on raw llm activations."""
        if self.is_folded.item():
            raise ValueError("Scaling factors already folded into weights")

        self._W_enc_XiDiL.mul_(scaling_factors_in_Xi[..., None, None])
        self._W_dec_LXoDo.div_(scaling_factors_out_Xo[..., None])
        if self._b_dec_XoDo is not None:
            self._b_dec_XoDo.div_(scaling_factors_out_Xo[..., None])

        # set buffer to prevent double-folding
        self.folded_scaling_factors_in_Xi = scaling_factors_in_Xi
        self.folded_scaling_factors_out_Xo = scaling_factors_out_Xo
        self.is_folded = torch.tensor(True, dtype=torch.bool)

    @torch.no_grad()
    def _with_folded_scaling_factors(
        self, scaling_factors_in_Xi: torch.Tensor, scaling_factors_out_Xo: torch.Tensor
    ) -> Self:
        cc = self.clone().to(self.device)
        cc.load_state_dict(self.state_dict())
        cc._fold_activation_scaling_into_weights_(scaling_factors_in_Xi, scaling_factors_out_Xo)
        return cc
