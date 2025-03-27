from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

import torch as t
from einops import einsum, reduce
from torch import nn

from model_diffing.models.activations import ACTIVATIONS_MAP, ActivationFunction
from model_diffing.saveable_module import SaveableModule
from model_diffing.utils import l2_norm, not_none

"""
Dimensions:
- B: batch size
- X: arbitrary number of crosscoding dimensions
- L: Latents, AKA "features"
- D: d_model
"""


TModel = TypeVar("TModel", bound=SaveableModule)


class InitStrategy(ABC, Generic[TModel]):
    @abstractmethod
    def init_weights(self, cc: TModel) -> None: ...


TActivation = TypeVar("TActivation", bound=ActivationFunction)


class AcausalCrosscoder(SaveableModule, Generic[TActivation]):
    is_folded: t.Tensor
    folded_scaling_factors_X: t.Tensor | None

    def __init__(
        self,
        crosscoding_dims: tuple[int, ...],
        d_model: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool | Literal["pre_bias"],
        skip_linear: bool = False,
        init_strategy: InitStrategy["AcausalCrosscoder[TActivation]"] | None = None,
        dtype: t.dtype = t.float32,
    ):
        super().__init__()
        self.crosscoding_dims = crosscoding_dims
        self.d_model = d_model
        self.n_latents = n_latents
        self.activation_fn = activation_fn
        self.dtype = dtype

        self.W_enc_XDL = nn.Parameter(t.empty((*crosscoding_dims, d_model, n_latents), dtype=dtype))
        self.W_dec_LXD = nn.Parameter(t.empty((n_latents, *crosscoding_dims, d_model), dtype=dtype))

        self.b_enc_L = nn.Parameter(t.empty((n_latents,), dtype=dtype)) if use_encoder_bias else None
        self.b_dec_XD = nn.Parameter(t.empty((*crosscoding_dims, d_model), dtype=dtype)) if use_decoder_bias else None
        self.pre_bias = use_decoder_bias == "pre_bias"

        self.W_skip_XDD = None
        if skip_linear:
            self.W_skip_XDD = nn.Parameter(t.empty((*crosscoding_dims, d_model, d_model), dtype=dtype))

        if init_strategy is not None:
            init_strategy.init_weights(self)

        # Initialize the buffer with a zero tensor of the correct shape, this means it's always serialized
        self.register_buffer("folded_scaling_factors_X", t.zeros(self.crosscoding_dims, dtype=dtype))
        # We also track whether it's actually holding a meaningful value by using this boolean flag.
        # Represented as a tensor so that it's serialized by torch.save
        self.register_buffer("is_folded", t.tensor(False, dtype=t.bool))

    @dataclass
    class ForwardResult:
        pre_activations_BL: t.Tensor
        latents_BL: t.Tensor
        recon_acts_BXD: t.Tensor

    def get_pre_bias_BL(self, activation_BXD: t.Tensor) -> t.Tensor:
        if self.pre_bias:
            activation_BXD -= not_none(self.b_dec_XD)
        return einsum(activation_BXD, self.W_enc_XDL, "b ..., ... l -> b l")

    def decode_BXD(self, latents_BL: t.Tensor) -> t.Tensor:
        pre_bias_BXD = einsum(latents_BL, self.W_dec_LXD, "b l, l ... d -> b ... d")
        if self.b_dec_XD:
            pre_bias_BXD += self.b_dec_XD
        return pre_bias_BXD

    def forward_train(
        self,
        activation_BXD: t.Tensor,
    ) -> ForwardResult:
        """returns the activations, the latents, and the reconstructed activations"""
        pre_activations_BL = self.get_pre_bias_BL(activation_BXD)

        if self.b_enc_L:
            pre_activations_BL += self.b_enc_L

        latents_BL = self.activation_fn.forward(pre_activations_BL)

        output_BXD = self.decode_BXD(latents_BL)

        if self.W_skip_XDD is not None:
            skip = einsum(activation_BXD, self.W_skip_XDD, "b ... d_in, ... d_in d_out -> b ... d_out")
            output_BXD += skip

        return self.ForwardResult(
            pre_activations_BL=pre_activations_BL,
            latents_BL=latents_BL,
            recon_acts_BXD=output_BXD,
        )

    def forward(self, activation_BXD: t.Tensor) -> t.Tensor:
        return self.forward_train(activation_BXD).recon_acts_BXD

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "crosscoding_dims": self.crosscoding_dims,
            "d_model": self.d_model,
            "n_latents": self.n_latents,
            "activation_classname": self.activation_fn.__class__.__name__,
            "activation_cfg": self.activation_fn._dump_cfg(),
            "use_encoder_bias": self.b_enc_L is not None,
            "use_decoder_bias": "pre_bias" if self.pre_bias else self.b_dec_XD is not None,
            # don't need to serialize init_strategy as loading from state_dict will re-initialize the params
        }

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "AcausalCrosscoder[TActivation]":
        activation_cfg = cfg["activation_cfg"]
        activation_classname = cfg["activation_classname"]

        activation_fn_cls = ACTIVATIONS_MAP[activation_classname]
        activation_fn = cast(TActivation, activation_fn_cls._from_cfg(activation_cfg))

        return AcausalCrosscoder(
            crosscoding_dims=cfg["crosscoding_dims"],
            d_model=cfg["d_model"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            # intentionally don't serialize init_strategy as loading from state_dict will re-initialize the params
        )

    @t.no_grad()
    def make_decoder_max_unit_norm_(self) -> None:
        """
        scales the decoder weights such that the model makes the same predictions, but for
        each latent, the maximum norm of it's decoder vectors is 1.

        For example, in a 2-model, 3-hookpoint crosscoder, the norms for a given latent might be scaled to:

        [[1, 0.2],
         [0.2, 0.4],
         [0.1, 0.3]]
        """
        output_space_norms_LX = reduce(self.W_dec_LXD, "l ... d -> l ...", l2_norm)
        assert output_space_norms_LX.shape[1:] == self.crosscoding_dims

        max_norms_per_latent_L = reduce(output_space_norms_LX, "l ... -> l", t.amax)
        assert max_norms_per_latent_L.shape == (self.n_latents,)

        # this means that the maximum norm of the decoder vectors into a given output space is 1
        # for example, in a cross-model cc, the norms for each model might be (1, 0.2) or (0.2, 1) or (1, 1)
        self.W_dec_LXD.copy_(einsum(self.W_dec_LXD, 1 / max_norms_per_latent_L, "l ..., l -> l ..."))
        self.W_enc_XDL.copy_(einsum(self.W_enc_XDL, max_norms_per_latent_L, "... l, l -> ... l"))

        if self.b_enc_L is not None:
            self.b_enc_L.copy_(self.b_enc_L * max_norms_per_latent_L)
        # no alteration needed for self.b_dec_XD

    def with_decoder_unit_norm(self) -> "AcausalCrosscoder[TActivation]":
        """
        returns a copy of the model with the weights rescaled such that the decoder norm of each feature is one,
        but the model makes the same predictions.
        """

        cc = AcausalCrosscoder(
            crosscoding_dims=self.crosscoding_dims,
            d_model=self.d_model,
            n_latents=self.n_latents,
            activation_fn=self.activation_fn,
            use_encoder_bias=self.b_enc_L is not None,
            use_decoder_bias=self.b_dec_XD is not None,
        )
        cc.to(self.W_dec_LXD.device)
        cc.load_state_dict(self.state_dict())
        cc.make_decoder_max_unit_norm_()

        return cc

    def fold_activation_scaling_into_weights_(self, scaling_factors_X: t.Tensor) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the m can be run on raw llm activations."""
        if self.is_folded.item():
            raise ValueError("Scaling factors already folded into weights")

        if scaling_factors_X.shape != self.crosscoding_dims:
            raise ValueError(f"Expected shape {self.crosscoding_dims}, got {scaling_factors_X.shape}")
        if t.any(scaling_factors_X == 0):
            raise ValueError("Scaling factors contain zeros, which would lead to division by zero")
        if t.any(t.isnan(scaling_factors_X)) or t.any(t.isinf(scaling_factors_X)):
            raise ValueError("Scaling factors contain NaN or Inf values")

        scaling_factors_X = scaling_factors_X.to(self.W_enc_XDL.device)

        self.W_enc_XDL.mul_(scaling_factors_X[..., None, None])
        self.W_dec_LXD.div_(scaling_factors_X[..., None])
        if self.b_dec_XD:
            self.b_dec_XD.div_(scaling_factors_X[..., None])

        # set buffer to prevent double-folding
        self.folded_scaling_factors_X = scaling_factors_X
        self.is_folded = t.tensor(True, dtype=t.bool)

    def with_folded_scaling_factors(self, scaling_factors_X: t.Tensor) -> "AcausalCrosscoder[TActivation]":
        cc = AcausalCrosscoder(
            crosscoding_dims=self.crosscoding_dims,
            d_model=self.d_model,
            n_latents=self.n_latents,
            activation_fn=self.activation_fn,
            use_encoder_bias=self.b_enc_L is not None,
            use_decoder_bias=self.b_dec_XD is not None,
        )
        cc.load_state_dict(self.state_dict())
        cc.fold_activation_scaling_into_weights_(scaling_factors_X)
        return cc