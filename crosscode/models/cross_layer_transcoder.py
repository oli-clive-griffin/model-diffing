from dataclasses import dataclass
from typing import Any, Generic, Self, cast

import torch
from einops import einsum
from torch import nn

from crosscode.models.activations import ACTIVATIONS_MAP
from crosscode.models.base_crosscoder import BaseCrosscoder, TActivation
from crosscode.models.initialization.init_strategy import InitStrategy
from crosscode.saveable_module import DTYPE_TO_STRING, STRING_TO_DTYPE


class CrossLayerTranscoder(Generic[TActivation], BaseCrosscoder[TActivation]):
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
        use_encoder_bias: bool = True,
        use_decoder_bias: bool = True,
        init_strategy: InitStrategy["CrossLayerTranscoder[TActivation]"] | None = None,
        dtype: torch.dtype = torch.float32,
        linear_skip: bool = False,
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
        self.d_model = d_model
        self.n_layers_out = n_layers_out

        self.W_skip_DPD = None
        if linear_skip:
            self.W_skip_DPD = nn.Parameter(torch.empty((d_model, n_layers_out, d_model), dtype=dtype))

        if init_strategy is not None:
            init_strategy.init_weights(self)

    @dataclass
    class ForwardResult:
        pre_activations_BL: torch.Tensor
        latents_BL: torch.Tensor
        output_BPD: torch.Tensor

    def forward_train(self, activation_BD: torch.Tensor) -> ForwardResult:
        res = self._forward_train(activation_BD)
        assert res.output_BXoDo.shape[1:-1] == (self.n_layers_out,)
        output_BPD = res.output_BXoDo

        if self.W_skip_DPD is not None:
            output_BPD += einsum(activation_BD, self.W_skip_DPD, "b d_in, d_in p d_out -> b p d_out")

        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            output_BPD=output_BPD,
        )

    def forward(self, activation_BD: torch.Tensor) -> torch.Tensor:
        return self.forward_train(activation_BD).output_BPD

    @property
    def W_dec_LPD(self) -> nn.Parameter:
        return self._W_dec_LXoDo

    @property
    def W_enc_DL(self) -> nn.Parameter:
        return self._W_enc_XiDiL

    @property
    def b_dec_PD(self) -> nn.Parameter | None:
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
            "linear_skip": self.W_skip_DPD is not None,
            "dtype": DTYPE_TO_STRING[self._dtype],
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
            linear_skip=cfg["linear_skip"],
            dtype=STRING_TO_DTYPE[cfg["dtype"]],
        )

    def with_folded_scaling_factors(self, scaling_factors_P: torch.Tensor) -> Self:
        scaling_factor_in = scaling_factors_P[0]
        scaling_factors_out_Po = scaling_factors_P[1:]
        return self._with_folded_scaling_factors(scaling_factor_in, scaling_factors_out_Po)