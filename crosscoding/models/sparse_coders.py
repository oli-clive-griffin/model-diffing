from dataclasses import dataclass
from typing import Any, Generic, Self, cast

import torch
from einops import einsum
from torch import nn

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
        crosscoding_dims = (n_models, len(hookpoints))

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
        self._crosscoding_dims = crosscoding_dims

        self.n_models = n_models
        self.hookpoints = hookpoints
        self.n_hookpoints = len(hookpoints)
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
            "n_models": self.n_models,
            "hookpoints": self.hookpoints,
            "d_model": self.d_model,
            "n_latents": self.n_latents,
            "activation_fn": {
                "classname": self.activation_fn.__class__.__name__,
                "cfg": self.activation_fn._dump_cfg(),
            },
            "use_encoder_bias": self.b_enc_L is not None,
            "use_decoder_bias": self._b_dec_XoDo is not None,
            "dtype": self._dtype,
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
        use_encoder_bias: bool,
        use_decoder_bias: bool,
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
            "dtype": self._dtype,
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
        n_layers_out: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool,
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
            self.W_skip_DPD = nn.Parameter(torch.empty((d_model, d_model), dtype=dtype))

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
            "n_layers_out": self.n_layers_out,
            "n_latents": self.n_latents,
            "activation_fn": {
                "classname": self.activation_fn.__class__.__name__,
                "cfg": self.activation_fn._dump_cfg(),
            },
            "use_encoder_bias": self.b_enc_L is not None,
            "use_decoder_bias": self._b_dec_XoDo is not None,
            "dtype": self._dtype,
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


class CompoundCrossLayerTranscoder(Generic[TActivation]):
    def __init__(
        self,
        hookpoints: list[str],
        d_model: int,
        n_latents_per_tc: int,
        activation_fn: TActivation,
        use_encoder_bias: bool,
        use_decoder_bias: bool,
        linear_skip: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_hookpoints = len(hookpoints)
        self.n_hookpoints_out = self.n_hookpoints - 1
        self.transcoders = nn.ModuleList(
            [
                CrossLayerTranscoder(
                    d_model=d_model,
                    n_layers_out=self.n_hookpoints_out - i,
                    n_latents=n_latents_per_tc,
                    activation_fn=activation_fn,
                    use_encoder_bias=use_encoder_bias,
                    use_decoder_bias=use_decoder_bias,
                    linear_skip=linear_skip,
                    dtype=dtype,
                )
                for i in range(len(hookpoints) - 1)
            ]
        )
        self.hookpoints = hookpoints
        self._dtype = dtype
        self.n_latents_per_tc = n_latents_per_tc

    @dataclass
    class ForwardResult:
        output_BPoD: torch.Tensor
        latents_BPoL: torch.Tensor
        pre_activations_BPoL: torch.Tensor

    def forward(self, activation_BPD: torch.Tensor) -> ForwardResult:
        # P = all hookpoints = len(self.hookpoints)
        # Po = out hookpoints = self.hookpoints[1:]
        B, P, D = activation_BPD.shape
        Po = P - 1
        device = activation_BPD.device
        dtype = activation_BPD.dtype
        output_BPoD = torch.zeros((B, Po, D), dtype=dtype, device=device)
        latents_BPoL = torch.zeros((B, Po, self.n_latents_per_tc), dtype=dtype, device=device)
        pre_activations_BPoL = torch.zeros((B, Po, self.n_latents_per_tc), dtype=dtype, device=device)

        for i in range(len(self.hookpoints) - 1):
            transcoder = cast(CrossLayerTranscoder[TActivation], self.transcoders[i])
            input_BD = activation_BPD[:, i]  # we should never take the last hookpoint as input

            res = transcoder.forward_train(input_BD)
            output_BPoD[:, i:, :] = res.output_BPD
            latents_BPoL[:, i, :] = res.latents_BL
            pre_activations_BPoL[:, i, :] = res.pre_activations_BL

        return self.ForwardResult(
            output_BPoD=output_BPoD,
            latents_BPoL=latents_BPoL,
            pre_activations_BPoL=pre_activations_BPoL,
        )
