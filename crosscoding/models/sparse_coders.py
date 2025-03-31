from dataclasses import dataclass
from typing import Any, Generic, Self, cast

import torch
from einops import einsum
from torch import nn

from crosscoding.dims import CrosscodingDim, CrosscodingDimsDict
from crosscoding.models.activations import ACTIVATIONS_MAP
from crosscoding.models.base_crosscoder import BaseCrosscoder, TActivation
from crosscoding.models.initialization.init_strategy import InitStrategy


class ModelHookpointAcausalCrosscoder(Generic[TActivation], BaseCrosscoder[TActivation]):
    def __init__(
        self,
        n_models: int,
        hookpoints: list[str],
        d_model: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool,
        init_strategy: InitStrategy["ModelHookpointAcausalCrosscoder[TActivation]"] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        crosscoding_dims = CrosscodingDimsDict.from_dims(
            CrosscodingDim(name="model", index_labels=[f"model_{i}" for i in range(n_models)]),
            CrosscodingDim(name="hookpoint", index_labels=hookpoints),
        )
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
        pre_activations_BL: torch.Tensor
        latents_BL: torch.Tensor
        recon_acts_BMPD: torch.Tensor

    def forward_train(self, activation_BMPD: torch.Tensor) -> ForwardResult:
        res = self._forward_train(activation_BMPD)
        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            recon_acts_BMPD=res.output_BXoDo,
        )

    def forward(self, activation_BMPD: torch.Tensor) -> torch.Tensor:
        return self.forward_train(activation_BMPD).recon_acts_BMPD

    def decode_BMPD(self, latents_BL: torch.Tensor) -> torch.Tensor:
        return self.decode_BXoDo(latents_BL)

    @property
    def W_dec_LMPD(self) -> torch.Tensor:
        return self._W_dec_LXoDo

    @property
    def W_enc_MPDL(self) -> torch.Tensor:
        return self._W_enc_XiDiL

    @property
    def b_dec_MPD(self) -> torch.Tensor | None:
        return self._b_dec_XoDo

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "n_models": len(self.crosscoding_dims["model"].index_labels),
            "hookpoints": self.crosscoding_dims["hookpoint"].index_labels,
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

        return ModelHookpointAcausalCrosscoder(
            n_models=cfg["n_models"],
            hookpoints=cfg["hookpoints"],
            d_model=cfg["d_model"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=cfg["dtype"],
        )

    # @classmethod
    # def create_as_SAE(
    #     cls: type[Self],
    #     d_model: int,
    #     n_latents: int,
    #     activation_fn: TActivation,
    #     use_encoder_bias: bool,
    #     use_decoder_bias: bool,
    #     dtype: torch.dtype = torch.float32,
    # ) -> "ModelHookpointAcausalCrosscoder[TActivation]":
    #     return ModelHookpointAcausalCrosscoder(
    #         crosscoding_dims=CrosscodingDimsDict(),
    #         d_model=d_model,
    #         n_latents=n_latents,
    #         activation_fn=activation_fn,
    #         use_encoder_bias=use_encoder_bias,
    #         use_decoder_bias=use_decoder_bias,
    #         dtype=dtype,
    #     )


class Transcoder(Generic[TActivation], BaseCrosscoder[TActivation]):
    # Xo = (),
    # Xi = (),
    # Di = D
    # Do = D

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool,
        init_strategy: InitStrategy["Transcoder[TActivation]"] | None = None,
        linear_skip: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            in_crosscoding_dims=CrosscodingDimsDict(),
            d_in=d_model,
            out_crosscoding_dims=CrosscodingDimsDict(),
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
            output_BD += einsum(output_BD, self.W_skip_DD, "b d, d d -> b d")

        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            output_BD=output_BD,
        )

    def forward(self, activation_BD: torch.Tensor) -> torch.Tensor:
        return self.forward_train(activation_BD).output_BD

    @property
    def W_dec_LD(self) -> torch.Tensor:
        return self._W_dec_LXoDo

    @property
    def W_enc_DL(self) -> torch.Tensor:
        return self._W_enc_XiDiL

    @property
    def b_dec_D(self) -> torch.Tensor | None:
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
            "dtype": self.dtype,
        }

    @classmethod
    def _scaffold_from_cfg(cls: type[Self], cfg: dict[str, Any]):
        activation = cfg["activation_fn"]
        activation_fn_cls = ACTIVATIONS_MAP[activation["classname"]]
        activation_fn = cast(TActivation, activation_fn_cls._scaffold_from_cfg(activation["cfg"]))

        return Transcoder(
            d_model=cfg["d_model"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=cfg["dtype"],
        )


class CrossLayerTranscoder(Generic[TActivation], BaseCrosscoder[TActivation]):
    # Xo = (P,)
    # Xi = (),
    # Di = D
    # Do = D

    def __init__(
        self,
        d_model: int,
        out_layers_names: list[str],
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool,
        init_strategy: InitStrategy["CrossLayerTranscoder[TActivation]"] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        out_layers_dim = CrosscodingDim(name="out_layer", index_labels=out_layers_names)
        super().__init__(
            in_crosscoding_dims=CrosscodingDimsDict(),
            d_in=d_model,
            out_crosscoding_dims=CrosscodingDimsDict.from_dims(out_layers_dim),
            d_out=d_model,
            n_latents=n_latents,
            activation_fn=activation_fn,
            use_encoder_bias=use_encoder_bias,
            use_decoder_bias=use_decoder_bias,
            init_strategy=None,
            dtype=dtype,
        )

        self.d_model = d_model
        self.out_layers_dim = out_layers_dim

        if init_strategy is not None:
            init_strategy.init_weights(self)

    @dataclass
    class ForwardResult:
        pre_activations_BL: torch.Tensor
        latents_BL: torch.Tensor
        output_BPD: torch.Tensor

    def forward_train(self, activation_BD: torch.Tensor) -> ForwardResult:
        res = self._forward_train(activation_BD)
        assert res.output_BXoDo.shape[1:-1] == self._out_crosscoding_dims.sizes()
        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            output_BPD=res.output_BXoDo,
        )

    def forward(self, activation_BD: torch.Tensor) -> torch.Tensor:
        return self.forward_train(activation_BD).output_BPD

    @property
    def W_dec_LPD(self) -> torch.Tensor:
        return self._W_dec_LXoDo

    @property
    def W_enc_DL(self) -> torch.Tensor:
        return self._W_enc_XiDiL

    @property
    def b_dec_PD(self) -> torch.Tensor | None:
        return self._b_dec_XoDo

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "out_layers_names": self._out_crosscoding_dims["out_layer"].index_labels,
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
            out_layers_names=cfg["out_layers_names"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=cfg["dtype"],
        )
