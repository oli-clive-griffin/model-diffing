from dataclasses import dataclass
from typing import Any, Generic, Self, cast

import torch
from torch import nn

from crosscode.models.activations import ACTIVATIONS_MAP
from crosscode.models.activations.relu import ReLUActivation
from crosscode.models.base_crosscoder import BaseCrosscoder, TActivation
from crosscode.models.initialization.init_strategy import InitStrategy


class ModelHookpointAcausalCrosscoder(Generic[TActivation], BaseCrosscoder[TActivation]):
    def __init__(
        self,
        n_models: int,
        n_hookpoints: int,
        d_model: int,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool = True,
        use_decoder_bias: bool = True,
        init_strategy: InitStrategy["ModelHookpointAcausalCrosscoder[TActivation]"] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        crosscoding_dims = (n_models, n_hookpoints)

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
        self.n_hookpoints = n_hookpoints
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
    def W_dec_LMPD(self) -> nn.Parameter:
        return self._W_dec_LXoDo

    @property
    def W_enc_MPDL(self) -> nn.Parameter:
        return self._W_enc_XiDiL

    @property
    def b_dec_MPD(self) -> nn.Parameter | None:
        return self._b_dec_XoDo

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "n_hookpoints": self.n_hookpoints,
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
            n_hookpoints=cfg["n_hookpoints"],
            d_model=cfg["d_model"],
            n_latents=cfg["n_latents"],
            activation_fn=activation_fn,
            use_encoder_bias=cfg["use_encoder_bias"],
            use_decoder_bias=cfg["use_decoder_bias"],
            dtype=cfg["dtype"],
        )

    def fold_activation_scaling_into_weights_(self, scaling_factors_out_MP: torch.Tensor) -> None:
        self._fold_activation_scaling_into_weights_(scaling_factors_out_MP, scaling_factors_out_MP)

    def with_folded_scaling_factors(self, scaling_factors_out_MP: torch.Tensor) -> Self:
        return self._with_folded_scaling_factors(scaling_factors_out_MP, scaling_factors_out_MP)


@dataclass
class ModelShape:
    hookpoint_dims: list[int]


@dataclass
class CrosscodingShape:
    models: list[ModelShape]


class IrregularModelHookpointAcausalCrosscoder(Generic[TActivation]):
    def __init__(
        self,
        shape: CrosscodingShape,
        n_latents: int,
        activation_fn: TActivation,
        use_encoder_bias: bool = True,
        use_decoder_bias: bool = True,
        init_strategy: InitStrategy["ModelHookpointAcausalCrosscoder[TActivation]"] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        max_d_model = max(hp_dim for model in shape.models for hp_dim in model.hookpoint_dims)
        max_n_hookpoints = max(len(model.hookpoint_dims) for model in shape.models)

        self.in_shape_padded_MPD = (len(shape.models), max_n_hookpoints, max_d_model)

        self.cc = ModelHookpointAcausalCrosscoder(
            n_models=len(shape.models),
            n_hookpoints=max_n_hookpoints,
            d_model=max_d_model,
            n_latents=n_latents,
            activation_fn=activation_fn,
            use_encoder_bias=use_encoder_bias,
            use_decoder_bias=use_decoder_bias,
            init_strategy=init_strategy,
            dtype=dtype,
        )
        self.shape = shape
        if init_strategy is not None:
            init_strategy.init_weights(self.cc)

    @dataclass
    class HookpointActivations:
        activations_BD: torch.Tensor

    @dataclass
    class ModelActivations:
        hookpoints: list["IrregularModelHookpointAcausalCrosscoder.HookpointActivations"]

    @dataclass
    class Activations:
        models: list["IrregularModelHookpointAcausalCrosscoder.ModelActivations"]
        batch_size: int

    @dataclass
    class ForwardResult:
        pre_activations_BL: torch.Tensor
        latents_BL: torch.Tensor
        recon_acts: "IrregularModelHookpointAcausalCrosscoder.Activations"

    def forward_train(self, activations: Activations) -> ForwardResult:
        padded_shape = (activations.batch_size, *self.in_shape_padded_MPD)
        in_BMPD = torch.zeros(padded_shape, dtype=self.cc._dtype, device=self.cc.device)
        assert len(activations.models) == len(self.shape.models)
        for model_idx, (model_acts, model_shape) in enumerate(zip(activations.models, self.shape.models, strict=True)):
            assert len(model_acts.hookpoints) == len(model_shape.hookpoint_dims)
            for hookpoint_idx, (hookpoint_acts, hookpoint_dim) in enumerate(
                zip(model_acts.hookpoints, model_shape.hookpoint_dims, strict=True)
            ):
                assert hookpoint_acts.activations_BD.shape[1] == hookpoint_dim
                in_BMPD[:, model_idx, hookpoint_idx, :hookpoint_dim] = hookpoint_acts.activations_BD

        res = self.cc.forward_train(in_BMPD)

        reconstructed_acts = self.Activations(
            models=[
                self.ModelActivations(
                    hookpoints=[
                        self.HookpointActivations(activations_BD=res.recon_acts_BMPD[:, model_idx, hookpoint_idx, :hookpoint_dim])
                        for hookpoint_idx, hookpoint_dim in enumerate(model_shape.hookpoint_dims)
                    ]
                )
                for model_idx, model_shape in enumerate(self.shape.models)
            ],
            batch_size=activations.batch_size,
        )

        return self.ForwardResult(
            pre_activations_BL=res.pre_activations_BL,
            latents_BL=res.latents_BL,
            recon_acts=reconstructed_acts,
        )

    def forward(self, activations: Activations) -> torch.Tensor:
        # MPDB
        output = self.forward_train(activations)
        return torch.nested.nested_tensor(
            [
                torch.nested.nested_tensor(
                    [
                        output.recon_acts.models[model_idx].hookpoints[hookpoint_idx].activations_BD
                        for hookpoint_idx in range(len(model.hookpoint_dims))
                    ]
                )
                for model_idx, model in enumerate(self.shape.models)
            ]
        )
        raise NotImplementedError()

