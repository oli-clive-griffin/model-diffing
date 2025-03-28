from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Self, TypeVar, cast

import einx
import torch as t
from einops import einsum, reduce
from torch import nn

from model_diffing.models.activations import ACTIVATIONS_MAP, ActivationFunction
from model_diffing.saveable_module import SaveableModule
from model_diffing.utils import l2_norm

"""
Dimensions:
- B: batch size
- Xi: arbitrary number of input crosscoding dimensions
- Di: input vector space dimensionality
- L: Latents, AKA "features"
- Xo: arbitrary number of output crosscoding dimensions
- Do: output vector space dimensionality
"""


TModel = TypeVar("TModel", bound=SaveableModule)


class InitStrategy(ABC, Generic[TModel]):
    @abstractmethod
    def init_weights(self, cc: TModel) -> None: ...


TActivation = TypeVar("TActivation", bound=ActivationFunction)


class _BaseCrosscoder(SaveableModule, Generic[TActivation]):
    is_folded: t.Tensor
    folded_scaling_factors_in_Xi: t.Tensor | None
    folded_scaling_factors_out_Xo: t.Tensor | None

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
        init_strategy: InitStrategy["_BaseCrosscoder[TActivation]"] | None = None,
        dtype: t.dtype = t.float32,
    ):
        super().__init__()
        self._in_crosscoding_dims = in_crosscoding_dims
        self._out_crosscoding_dims = out_crosscoding_dims

        self.n_latents = n_latents
        self.activation_fn = activation_fn
        self.dtype = dtype

        self._W_enc_XiDiL = nn.Parameter(t.empty((*in_crosscoding_dims, d_in, n_latents), dtype=dtype))
        self._W_dec_LXoDo = nn.Parameter(t.empty((n_latents, *out_crosscoding_dims, d_out), dtype=dtype))

        # public because no implementations rename it
        self.b_enc_L = nn.Parameter(t.empty((n_latents,), dtype=dtype)) if use_encoder_bias else None
        self._b_dec_XoDo = (
            nn.Parameter(t.empty((*out_crosscoding_dims, d_out), dtype=dtype)) if use_decoder_bias else None
        )

        if init_strategy is not None:
            init_strategy.init_weights(self)

        # Initialize buffers with a tensors of the correct shapes, this means it's always serialized
        self.register_buffer("folded_scaling_factors_in_Xi", t.zeros(self._in_crosscoding_dims, dtype=dtype))
        self.register_buffer("folded_scaling_factors_out_Xo", t.zeros(self._out_crosscoding_dims, dtype=dtype))
        # We also track whether it's actually holding a meaningful value by using this boolean flag.
        # Represented as a tensor so that it's serialized by torch.save
        self.register_buffer("is_folded", t.tensor(False, dtype=t.bool))

    @dataclass
    class _ForwardResult:
        pre_activations_BL: t.Tensor
        latents_BL: t.Tensor
        output_BXoDo: t.Tensor

    def get_pre_bias_BL(self, activation_BXiDi: t.Tensor) -> t.Tensor:
        return einsum(activation_BXiDi, self._W_enc_XiDiL, "b ..., ... l -> b l")

    def decode_BXoDo(self, latents_BL: t.Tensor) -> t.Tensor:
        pre_bias_BXoDo = einsum(latents_BL, self._W_dec_LXoDo, "b l, l ... -> b ...")
        if self._b_dec_XoDo is not None:
            pre_bias_BXoDo += self._b_dec_XoDo
        return pre_bias_BXoDo

    def _forward_train(
        self,
        activation_BXiDi: t.Tensor,
    ) -> _ForwardResult:
        """returns the activations, the latents, and the reconstructed activations"""
        pre_activations_BL = self.get_pre_bias_BL(activation_BXiDi)

        if self.b_enc_L is not None:
            pre_activations_BL += self.b_enc_L

        latents_BL = self.activation_fn.forward(pre_activations_BL)

        output_BXoDo = self.decode_BXoDo(latents_BL)

        return self._ForwardResult(
            pre_activations_BL=pre_activations_BL,
            latents_BL=latents_BL,
            output_BXoDo=output_BXoDo,
        )

    # def forward(self, activation_BXiDi: t.Tensor) -> t.Tensor:
    #     return self.forward_train(activation_BXiDi).output_BXoDo

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
        # output_space_norms_LX = reduce(self.W_dec_LXoDo, "l ... do -> l ...", l2_norm)
        output_space_norms_LX = reduce(self._W_dec_LXoDo, "l ... do -> l ...", l2_norm)
        output_space_norms_LX_ = einx.reduce("... [do]", self._W_dec_LXoDo, l2_norm)
        assert t.allclose(output_space_norms_LX, output_space_norms_LX_)

        max_norms_per_latent_L = reduce(output_space_norms_LX, "l ... -> l", t.amax)
        max_norms_per_latent_L_ = einx.reduce("l [...]", output_space_norms_LX, t.amax)
        assert t.allclose(max_norms_per_latent_L, max_norms_per_latent_L_)

        # this means that the maximum norm of the decoder vectors into a given output space is 1
        # for example, in a cross-model cc, the norms for each model might be (1, 0.2) or (0.2, 1) or (1, 1)
        self._W_dec_LXoDo.copy_(einsum(self._W_dec_LXoDo, 1 / max_norms_per_latent_L, "l ..., l -> l ..."))
        self._W_enc_XiDiL.copy_(einsum(self._W_enc_XiDiL, max_norms_per_latent_L, "... l, l -> ... l"))

        if self.b_enc_L is not None:
            self.b_enc_L.copy_(self.b_enc_L * max_norms_per_latent_L)
        # no alteration needed for self.b_dec_XoDo

    @property
    def device(self) -> t.device:
        return self._W_dec_LXoDo.device

    def clone(self) -> Self:
        out = self._scaffold_from_cfg(self._dump_cfg())
        out.load_state_dict(self.state_dict())
        return out

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
        self, scaling_factors_in_Xi: t.Tensor, scaling_factors_out_Xo: t.Tensor
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
        self.is_folded = t.tensor(True, dtype=t.bool)

    def with_folded_scaling_factors(self, scaling_factors_in_Xi: t.Tensor, scaling_factors_out_Xo: t.Tensor) -> Self:
        cc = self.clone().to(self.device)
        cc.load_state_dict(self.state_dict())
        cc.fold_activation_scaling_into_weights_(scaling_factors_in_Xi, scaling_factors_out_Xo)
        return cc


class AcausalCrosscoder(_BaseCrosscoder[TActivation], Generic[TActivation]):
    def __init__(
        self,
        crosscoding_dims: tuple[int, ...],
        d_model: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool,
        init_strategy: InitStrategy["AcausalCrosscoder[TActivation]"] | None = None,
        dtype: t.dtype = t.float32,
    ):
        super().__init__(
            crosscoding_dims,
            d_model,
            crosscoding_dims,
            d_model,
            n_latents,
            activation_fn,
            use_encoder_bias,
            use_decoder_bias,
            None,
            dtype,
        )
        self.crosscoding_dims = crosscoding_dims
        self.d_model = d_model
        if init_strategy is not None:
            init_strategy.init_weights(self)

    @dataclass
    class ForwardResult:
        pre_activations_BL: t.Tensor
        latents_BL: t.Tensor
        recon_acts_BXD: t.Tensor

    def forward_train(self, activation_BXD: t.Tensor) -> ForwardResult:
        res = self._forward_train(activation_BXD)
        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            recon_acts_BXD=res.output_BXoDo,
        )

    def forward(self, activation_BXD: t.Tensor) -> t.Tensor:
        return self.forward_train(activation_BXD).recon_acts_BXD

    def decode_BXD(self, latents_BL: t.Tensor) -> t.Tensor:
        return self.decode_BXoDo(latents_BL)

    @property
    def W_dec_LXD(self) -> t.Tensor:
        return self._W_dec_LXoDo

    @property
    def W_enc_XDL(self) -> t.Tensor:
        return self._W_enc_XiDiL

    @property
    def b_dec_XD(self) -> t.Tensor | None:
        return self._b_dec_XoDo

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "crosscoding_dims": self.crosscoding_dims,
            "d_model": self.d_model,
            "n_latents": self.n_latents,
            "activation_fn": {
                "classname": self.activation_fn.__class__.__name__,
                "cfg": self.activation_fn._dump_cfg(),
            },
            "use_encoder_bias": self.b_enc_L is not None,
            "use_decoder_bias": self._b_dec_XoDo is not None,
            "dtype": self.dtype,
        }

    @classmethod
    def _scaffold_from_cfg(cls: type[Self], cfg: dict[str, Any]):
        activation = cfg["activation_fn"]
        activation_fn_cls = ACTIVATIONS_MAP[activation["classname"]]
        activation_fn = cast(TActivation, activation_fn_cls._scaffold_from_cfg(activation["cfg"]))

        return AcausalCrosscoder(
            crosscoding_dims=cfg["crosscoding_dims"],
            d_model=cfg["d_model"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=cfg["dtype"],
        )

    # def _dump_cfg(self) -> dict[str, Any]:
    #     return {
    #         "in_crosscoding_dims": self._in_crosscoding_dims,
    #         "d_in": self._d_in,
    #         "out_crosscoding_dims": self._out_crosscoding_dims,
    #         "d_out": self._d_out,
    #         "n_latents": self.n_latents,
    #         "activation_classname": self.activation_fn.__class__.__name__,
    #         "activation_cfg": self.activation_fn._dump_cfg(),
    #         "use_encoder_bias": self.b_enc_L is not None,
    #         "use_decoder_bias": self._b_dec_XoDo is not None,
    #         # don't need to serialize init_strategy as loading from state_dict will re-initialize the params
    #     }

    # @classmethod
    # def _from_cfg(cls, cfg: dict[str, Any]) -> "BaseCrosscoder[TActivation]":
    #     activation_cfg = cfg["activation_cfg"]
    #     activation_classname = cfg["activation_classname"]

    #     activation_fn_cls = ACTIVATIONS_MAP[activation_classname]
    #     activation_fn = cast(TActivation, activation_fn_cls._from_cfg(activation_cfg))

    #     return BaseCrosscoder(
    #         in_crosscoding_dims=cfg["in_crosscoding_dims"],
    #         d_in=cfg["d_in"],
    #         out_crosscoding_dims=cfg["out_crosscoding_dims"],
    #         d_out=cfg["d_out"],
    #         n_latents=cfg["n_latents"],
    #         activation_fn=activation_fn,
    #         use_encoder_bias=cfg["use_encoder_bias"],
    #         use_decoder_bias=cfg["use_decoder_bias"],
    #         # intentionally don't serialize init_strategy as loading from state_dict will re-initialize the params
    #     )


class CrossLayerTranscoder(_BaseCrosscoder[TActivation], Generic[TActivation]):
    # Xo = (P,)
    # Xi = (),
    # Di = D
    # Do = D

    def __init__(
        self,
        d_model: int,
        n_layers_out: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool,
        init_strategy: InitStrategy["CrossLayerTranscoder[TActivation]"] | None = None,
        dtype: t.dtype = t.float32,
    ):
        super().__init__(
            in_crosscoding_dims=(),
            d_in=d_model,
            out_crosscoding_dims=(n_layers_out,),
            d_out=d_model,
            n_latents=n_latents,
            activation_fn=activation_fn,
            use_encoder_bias=use_encoder_bias,
            use_decoder_bias=use_decoder_bias,
            init_strategy=None,
            dtype=dtype,
        )

        self.n_layers_out = n_layers_out
        self.d_model = d_model

        if init_strategy is not None:
            init_strategy.init_weights(self)

    @dataclass
    class ForwardResult:
        pre_activations_BL: t.Tensor
        latents_BL: t.Tensor
        output_BPD: t.Tensor

    def forward_train(self, activation_BD: t.Tensor) -> ForwardResult:
        # valid becuase Xi = `()`, so BXiDi == (B, D)
        res = self._forward_train(activation_BD)
        assert res.output_BXoDo.shape[1:] == (self.n_layers_out, self.d_model)
        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            output_BPD=res.output_BXoDo,
        )

    def forward(self, activation_BD: t.Tensor) -> t.Tensor:
        return self.forward_train(activation_BD).output_BPD

    @property
    def W_dec_LPD(self) -> t.Tensor:
        return self._W_dec_LXoDo

    @property
    def W_enc_DL(self) -> t.Tensor:
        return self._W_enc_XiDiL

    @property
    def b_dec_PD(self) -> t.Tensor | None:
        return self._b_dec_XoDo

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_layers_out": self.n_layers_out,
            "n_latents": self.n_latents,
            "activation_fn": {
                "classname": self.activation_fn.__class__.__name__,
                "cfg": self.activation_fn._dump_cfg(),
            },
            "use_encoder_bias": self.b_enc_L is not None,
            "use_decoder_bias": self._b_dec_XoDo is not None,
            "dtype": self.dtype,
        }

    @classmethod
    def _scaffold_from_cfg(cls: type[Self], cfg: dict[str, Any]):
        activation = cfg["activation_fn"]
        activation_fn_cls = ACTIVATIONS_MAP[activation["classname"]]
        activation_fn = cast(TActivation, activation_fn_cls._scaffold_from_cfg(activation["cfg"]))

        return CrossLayerTranscoder(
            d_model=cfg["d_model"],
            n_layers_out=cfg["n_layers_out"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=cfg["dtype"],
        )
