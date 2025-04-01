from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import einx
import torch
from einops import einsum, reduce
from torch import nn

from crosscoding.models.activations import ActivationFunction
from crosscoding.models.initialization.init_strategy import InitStrategy
from crosscoding.saveable_module import SaveableModule
from crosscoding.utils import l2_norm

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
        use_encoder_bias: bool,
        use_decoder_bias: bool,
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

        pre_activations_BL = self.get_pre_bias_BL(activation_BXiDi)

        if self.b_enc_L is not None:
            pre_activations_BL += self.b_enc_L

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
    def make_decoder_max_unit_norm_(self) -> None:
        """
        scales the decoder weights such that the model makes the same predictions, but for
        each latent, the maximum norm of it's decoder vectors is 1.

        For example, in a 2-model, 3-hookpoint crosscoder, the norms for a given latent might be scaled to:

        [[1, 0.2],
         [0.2, 0.4],
         [0.1, 0.3]]
        """
        # output_space_norms_LX = reduce(self.W_dec_LXoDo, "l ... do -> l ...", l2_norm)
        output_space_norms_LX = reduce(self._W_dec_LXoDo, "l ... do -> l ...", l2_norm)
        output_space_norms_LX_ = einx.reduce("... [do]", self._W_dec_LXoDo, l2_norm)
        assert torch.allclose(output_space_norms_LX, output_space_norms_LX_)

        max_norms_per_latent_L = reduce(output_space_norms_LX, "l ... -> l", torch.amax)
        max_norms_per_latent_L_ = einx.reduce("l [...]", output_space_norms_LX, torch.amax)
        assert torch.allclose(max_norms_per_latent_L, max_norms_per_latent_L_)

        # this means that the maximum norm of the decoder vectors into a given output space is 1
        # for example, in a cross-model cc, the norms for each model might be (1, 0.2) or (0.2, 1) or (1, 1)
        self._W_dec_LXoDo.copy_(einsum(self._W_dec_LXoDo, 1 / max_norms_per_latent_L, "l ..., l -> l ..."))
        self._W_enc_XiDiL.copy_(einsum(self._W_enc_XiDiL, max_norms_per_latent_L, "... l, l -> ... l"))

        if self.b_enc_L is not None:
            self.b_enc_L.copy_(self.b_enc_L * max_norms_per_latent_L)
        # no alteration needed for self.b_dec_XoDo

    def with_decoder_unit_norm(self) -> Self:
        """
        returns a copy of the model with the weights rescaled such that the decoder norm of each feature is one,
        but the model makes the same predictions.
        """

        cc = self.clone().to(self.device)
        cc.load_state_dict(self.state_dict())
        cc.make_decoder_max_unit_norm_()

        return cc

    def fold_activation_scaling_into_weights_(
        self, scaling_factors_in_Xi: torch.Tensor, scaling_factors_out_Xo: torch.Tensor
    ) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the m can be run on raw llm activations."""
        if self.is_folded.item():
            raise ValueError("Scaling factors already folded into weights")

        raise NotImplementedError("Not tested")
        self._W_enc_XiDiL.mul_(scaling_factors_in_Xi[..., None, None])
        self._W_dec_LXoDo.div_(scaling_factors_out_Xo[..., None])
        if self._b_dec_XoDo:
            self._b_dec_XoDo.div_(scaling_factors_out_Xo[..., None])

        # set buffer to prevent double-folding
        self.folded_scaling_factors_in_Xi = scaling_factors_in_Xi
        self.folded_scaling_factors_out_Xo = scaling_factors_out_Xo
        self.is_folded = torch.tensor(True, dtype=torch.bool)

    def with_folded_scaling_factors(
        self, scaling_factors_in_Xi: torch.Tensor, scaling_factors_out_Xo: torch.Tensor
    ) -> Self:
        cc = self.clone().to(self.device)
        cc.load_state_dict(self.state_dict())
        cc.fold_activation_scaling_into_weights_(scaling_factors_in_Xi, scaling_factors_out_Xo)
        return cc
