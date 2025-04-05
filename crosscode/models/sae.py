from dataclasses import dataclass
from typing import Any, Generic, Self, cast

import torch
from einops import einsum
from torch import nn

from crosscode.models.activations import ACTIVATIONS_MAP
from crosscode.models.base_crosscoder import BaseCrosscoder, TActivation
from crosscode.models.initialization.init_strategy import InitStrategy
from crosscode.saveable_module import DTYPE_TO_STRING, STRING_TO_DTYPE


class SAEOrTranscoder(Generic[TActivation], BaseCrosscoder[TActivation]):
    # Xo = (),
    # Xi = (),
    # Di = D
    # Do = D

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool = True,
        use_decoder_bias: bool = True,
        init_strategy: InitStrategy["SAEOrTranscoder[TActivation]"] | None = None,
        linear_skip: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            in_crosscoding_dims=(),
            d_in=d_model,
            out_crosscoding_dims=(),
            d_out=d_model,
            n_latents=n_latents,
            activation_fn=activation_fn,
            use_encoder_bias=use_encoder_bias,
            use_decoder_bias=use_decoder_bias,
            init_strategy=None,
            dtype=dtype,
        )

        self.d_model = d_model
        self.W_skip_DD = nn.Parameter(torch.empty((d_model, d_model), dtype=dtype)) if linear_skip else None

        if init_strategy is not None:
            init_strategy.init_weights(self)

    @dataclass
    class ForwardResult:
        pre_activations_BL: torch.Tensor
        latents_BL: torch.Tensor
        output_BD: torch.Tensor

    def forward_train(self, activation_BD: torch.Tensor) -> ForwardResult:
        res = self._forward_train(activation_BD)

        assert res.output_BXoDo.shape[1:] == (self.d_model,)  # no crosscoding dims

        output_BD = res.output_BXoDo
        if self.W_skip_DD is not None:
            output_BD += einsum(activation_BD, self.W_skip_DD, "b d_in, d_in d_out -> b d_out")

        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            output_BD=output_BD,
        )

    def forward(self, activation_BD: torch.Tensor) -> torch.Tensor:
        return self.forward_train(activation_BD).output_BD

    @property
    def W_dec_LD(self) -> nn.Parameter:
        return self._W_dec_LXoDo

    @property
    def W_enc_DL(self) -> nn.Parameter:
        return self._W_enc_XiDiL

    @property
    def b_dec_D(self) -> nn.Parameter | None:
        return self._b_dec_XoDo

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_latents": self.n_latents,
            "activation_fn": {
                "classname": self.activation_fn.__class__.__name__,
                "cfg": self.activation_fn._dump_cfg(),
            },
            "use_encoder_bias": self.b_enc_L is not None,
            "use_decoder_bias": self._b_dec_XoDo is not None,
            "dtype": DTYPE_TO_STRING[self._dtype],
        }

    @classmethod
    def _scaffold_from_cfg(cls: type[Self], cfg: dict[str, Any]):
        activation = cfg["activation_fn"]
        activation_fn_cls = ACTIVATIONS_MAP[activation["classname"]]
        activation_fn = cast(TActivation, activation_fn_cls._scaffold_from_cfg(activation["cfg"]))

        return SAEOrTranscoder(
            d_model=cfg["d_model"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=STRING_TO_DTYPE[cfg["dtype"]],
        )

    def fold_activation_scaling_into_weights_(self, scaling_factor: float) -> None:
        scaling_factor_tensor = torch.tensor(scaling_factor)
        self._fold_activation_scaling_into_weights_(scaling_factor_tensor, scaling_factor_tensor)

    def with_folded_scaling_factors(self, scaling_factor: float) -> Self:
        scaling_factor_tensor = torch.tensor(scaling_factor)
        return self._with_folded_scaling_factors(scaling_factor_tensor, scaling_factor_tensor)
