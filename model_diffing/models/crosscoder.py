from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import torch as t
from einops import einsum, rearrange, reduce
from torch import nn

from model_diffing.models.activations import ACTIVATIONS
from model_diffing.utils import SaveableModule, l2_norm

"""
Dimensions:
- B: batch size
- X: arbitrary number of crosscoding dimensions
- H: Autoencoder h dimension
- D: Autoencoder d dimension
"""

TActivation = TypeVar("TActivation", bound=SaveableModule)


class AcausalCrosscoder(SaveableModule, Generic[TActivation]):
    is_folded: t.Tensor
    folded_scaling_factors_X: t.Tensor | None

    def __init__(
        self,
        crosscoding_dims: tuple[int, ...],
        d_model: int,
        hidden_dim: int,
        dec_init_norm: float,
        hidden_activation: TActivation,
    ):
        super().__init__()
        self.crosscoding_dims = crosscoding_dims
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

        W_base_HXD = t.randn((hidden_dim, *crosscoding_dims, d_model))

        self.W_dec_HXD = nn.Parameter(W_base_HXD.clone())

        with t.no_grad():
            W_dec_norm_HX1 = l2_norm(self.W_dec_HXD, dim=-1, keepdim=True)
            self.W_dec_HXD.div_(W_dec_norm_HX1)
            self.W_dec_HXD.mul_(dec_init_norm)

            self.W_enc_XDH = nn.Parameter(
                rearrange(  # "transpose" of the encoder weights
                    W_base_HXD,
                    "h ... -> ... h",
                )
            )

        self.b_dec_XD = nn.Parameter(t.zeros((*crosscoding_dims, d_model)))
        self.b_enc_H = nn.Parameter(t.zeros((hidden_dim,)))

        # Initialize the buffer with a zero tensor of the correct shape
        self.register_buffer("folded_scaling_factors_X", None)
        # track this boolean flag as a tensor so that it's serialized by torch.save
        self.register_buffer("is_folded", t.tensor(False, dtype=t.bool))

    def _encode_BH(self, activation_BXD: t.Tensor) -> t.Tensor:
        pre_bias_BH = einsum(
            activation_BXD,
            self.W_enc_XDH,
            "b ..., ... h -> b h",
        )
        pre_activation_BH = pre_bias_BH + self.b_enc_H
        return self.hidden_activation(pre_activation_BH)

    def _decode_BXD(self, hidden_BH: t.Tensor) -> t.Tensor:
        pre_bias_BXD = einsum(hidden_BH, self.W_dec_HXD, "b h, h ... d -> b ... d")
        return pre_bias_BXD + self.b_dec_XD

    @dataclass
    class TrainResult:
        hidden_BH: t.Tensor
        reconstructed_acts_BXD: t.Tensor

    def forward_train(
        self,
        activation_BXD: t.Tensor,
    ) -> TrainResult:
        """returns the activations, the h states, and the reconstructed activations"""
        self._validate_acts_shape(activation_BXD)

        hidden_BH = self._encode_BH(activation_BXD)
        reconstructed_BXD = self._decode_BXD(hidden_BH)

        if reconstructed_BXD.shape != activation_BXD.shape:
            raise ValueError(
                f"reconstructed_BXD.shape {reconstructed_BXD.shape} != activation_BXD.shape {activation_BXD.shape}"
            )

        return self.TrainResult(
            hidden_BH=hidden_BH,
            reconstructed_acts_BXD=reconstructed_BXD,
        )

    def _validate_acts_shape(self, activation_BXD: t.Tensor) -> None:
        _, *crosscoding_dims, d_model = activation_BXD.shape
        if d_model != self.d_model:
            raise ValueError(f"activation_BXD.shape[-1] {d_model} != d_model {self.d_model}")

        if crosscoding_dims != list(self.crosscoding_dims):
            raise ValueError(
                f"activation_BXD.shape[:-1] {crosscoding_dims} != crosscoding_dims {self.crosscoding_dims}"
            )

    def forward(self, activation_BXD: t.Tensor) -> t.Tensor:
        hidden_BH = self._encode_BH(activation_BXD)
        return self._decode_BXD(hidden_BH)

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "crosscoding_dims": self.crosscoding_dims,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "hidden_activation_classname": self.hidden_activation.__class__.__name__,
            "hidden_activation_cfg": self.hidden_activation._dump_cfg(),
        }

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "AcausalCrosscoder[TActivation]":
        hidden_activation_cfg = cfg["hidden_activation_cfg"]
        hidden_activation_classname = cfg["hidden_activation_classname"]

        hidden_activation_cls = ACTIVATIONS[hidden_activation_classname]
        hidden_activation = cast(TActivation, hidden_activation_cls._from_cfg(hidden_activation_cfg))

        return AcausalCrosscoder(
            crosscoding_dims=cfg["crosscoding_dims"],
            d_model=cfg["d_model"],
            hidden_dim=cfg["hidden_dim"],
            dec_init_norm=0,  # dec_init_norm doesn't matter here as we should be loading weights from a checkpoint
            hidden_activation=hidden_activation,
        )

    def with_decoder_unit_norm(self) -> "AcausalCrosscoder[TActivation]":
        """
        returns a copy of the model with the weights rescaled such that the decoder norm of each feature is one,
        but the model makes the same predictions.
        """

        cc = AcausalCrosscoder(
            crosscoding_dims=self.crosscoding_dims,
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            dec_init_norm=0,  # this shouldn't matter
            hidden_activation=self.hidden_activation,
        )

        with t.no_grad():

            class ImPrettySureThisIsIncorrect(Exception):
                pass

            raise ImPrettySureThisIsIncorrect(
                "need to figure out how the math works in the arbitrary extra dimensions case"
            )

            W_dec_l2_norms_X1H = reduce(self.W_dec_HXD, "h ... d -> ... 1 h", l2_norm)
            W_dec_l2_norms_H = reduce(self.W_dec_HXD, "h ... -> h", l2_norm)
            W_dec_l2_norms_HX1 = reduce(self.W_dec_HXD, "h ... d -> h ... 1", l2_norm)

            cc.W_enc_XDH.copy_(self.W_enc_XDH * W_dec_l2_norms_X1H)
            cc.b_enc_H.copy_(self.b_enc_H * W_dec_l2_norms_H)
            cc.W_dec_HXD.copy_(self.W_dec_HXD / W_dec_l2_norms_HX1)
            # no alteration needed for self.b_dec_XD

        return cc

    @t.no_grad()
    def _scale_weights(self, scaling_factors_X: t.Tensor) -> None:
        """the shape of scaling_factors_X must be the same as self.crosscoding_dims"""
        if scaling_factors_X.shape != self.crosscoding_dims:
            raise ValueError(
                f"scaling_factors_X.shape {scaling_factors_X.shape} != self.crosscoding_dims {self.crosscoding_dims}"
            )

        self.W_enc_XDH.mul_(scaling_factors_X[..., None, None])
        self.W_dec_HXD.div_(scaling_factors_X[..., None])
        self.b_dec_XD.div_(scaling_factors_X[..., None])

    def _validate_scaling_factors(self, scaling_factors_X: t.Tensor) -> None:
        if scaling_factors_X.shape != self.crosscoding_dims:
            raise ValueError(f"Expected shape {self.crosscoding_dims}, got {scaling_factors_X.shape}")
        if t.any(scaling_factors_X == 0):
            raise ValueError("Scaling factors contain zeros, which would lead to division by zero")
        if t.any(t.isnan(scaling_factors_X)) or t.any(t.isinf(scaling_factors_X)):
            raise ValueError("Scaling factors contain NaN or Inf values")

    @contextmanager
    def temporarily_fold_activation_scaling(self, scaling_factors_X: t.Tensor):
        """Temporarily fold scaling factors into weights."""
        self.fold_activation_scaling_into_weights_(scaling_factors_X)
        yield
        _ = self.unfold_activation_scaling_from_weights_()

    def fold_activation_scaling_into_weights_(self, scaling_factors_X: t.Tensor) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the m can be run on raw llm activations."""
        if self.is_folded.item():
            raise ValueError("Scaling factors already folded into weights")

        self._validate_scaling_factors(scaling_factors_X)
        scaling_factors_X = scaling_factors_X.to(self.W_enc_XDH.device)
        self._scale_weights(scaling_factors_X)

        # set buffer to prevent double-folding
        self.folded_scaling_factors_X = scaling_factors_X
        self.is_folded = t.tensor(True, dtype=t.bool)

    def unfold_activation_scaling_from_weights_(self) -> t.Tensor:
        if not self.is_folded.item():
            raise ValueError("No folded scaling factors found")

        if self.folded_scaling_factors_X is None:
            raise ValueError("Inconsistent state: is_folded is True but folded_scaling_factors_X is None")

        folded_scaling_factors_X = self.folded_scaling_factors_X.clone()
        # Clear the buffer before operations to prevent double-unfolding

        self.folded_scaling_factors_X = None
        self.is_folded = t.tensor(False, dtype=t.bool)

        self._scale_weights(1 / folded_scaling_factors_X)

        return folded_scaling_factors_X
