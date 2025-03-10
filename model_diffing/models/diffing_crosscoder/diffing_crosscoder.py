from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import torch as t
from einops import einsum, reduce, repeat
from torch import nn

from model_diffing.log import logger
from model_diffing.models import InitStrategy
from model_diffing.models.activations import ACTIVATIONS_MAP
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.utils import SaveableModule, l2_norm, runtimecast

"""
Dimensions:
- B: batch size
- M: number of models
- H: Autoencoder h dimension
- D: Autoencoder d dimension
"""

TActivation = TypeVar("TActivation", bound=ActivationFunction)


N_MODELS = 2


class DiffingCrosscoder(SaveableModule, Generic[TActivation]):
    """Model-Diffing Crosscoder from"""

    is_folded: t.Tensor
    folded_scaling_factors_M: t.Tensor | None

    def __init__(
        self,
        d_model: int,
        n_latents_total: int,
        n_shared_latents: int,
        activation_fn: TActivation,
        init_strategy: InitStrategy["DiffingCrosscoder[TActivation]"] | None = None,
        dtype: t.dtype = t.float32,
    ):
        super().__init__()

        self.d_model = d_model

        assert n_shared_latents <= n_latents_total
        if n_shared_latents > n_latents_total / 10:
            logger.warning(
                f"n_shared_latents {n_shared_latents} is greater than 10% of n_latents_total {n_latents_total}, are you sure this was intentional?"
            )

        self.n_latents_total = n_latents_total
        self.n_shared_latents = n_shared_latents

        self.W_enc_MDH = nn.Parameter(t.empty((N_MODELS, d_model, n_latents_total), dtype=dtype))
        self.b_enc_H = nn.Parameter(t.empty((n_latents_total,), dtype=dtype))

        self.activation_fn = activation_fn

        self._W_dec_shared_m0_HsD = nn.Parameter(t.empty((n_shared_latents, d_model), dtype=dtype))
        self._W_dec_shared_m1_HsD = self._W_dec_shared_m0_HsD
        """decoder weights for the shared features"""

        self._W_dec_indep_HiMD = nn.Parameter(t.empty((n_latents_total - n_shared_latents, N_MODELS, d_model), dtype=dtype))
        """decoder weights for the model-specific features"""

        self.b_dec_MD = nn.Parameter(t.empty((N_MODELS, d_model), dtype=dtype))

        if init_strategy is not None:
            init_strategy.init_weights(self)

        # Initialize the buffer with a zero tensor of the correct shape, this means it's always serialized
        self.register_buffer("folded_scaling_factors_M", t.zeros(N_MODELS, dtype=dtype))
        # However, track whether it's actually holding a meaningful value by using this boolean flag.
        # Represented as a tensor so that it's serialized by torch.save
        self.register_buffer("is_folded", t.tensor(False, dtype=t.bool))

    def theoretical_decoder_W_dec_HMD(self) -> t.Tensor:
        """
        This is not a real weight matrix, it's theoretically what the decoder WOULD be without the weight sharing
        """
        W_dec_shared_HsMD = t.stack([self._W_dec_shared_m0_HsD, self._W_dec_shared_m1_HsD], dim=1)

        # IMPORTANT: shared latents are the FIRST `n_shared_latents` along the hidden dim, ordering matters
        W_dec_HMD = self._join_shared_indep(shared=W_dec_shared_HsMD, indep=self._W_dec_indep_HiMD, dim=0)
        assert W_dec_HMD.shape == (self.n_latents_total, N_MODELS, self.d_model)

        return W_dec_HMD

    @dataclass
    class ForwardResult:
        hidden_shared_BHs: t.Tensor
        hidden_indep_BHi: t.Tensor
        recon_acts_BMD: t.Tensor

        def get_hidden_BH(self) -> t.Tensor:
            return t.cat([self.hidden_shared_BHs, self.hidden_indep_BHi], dim=-1)

    def forward_train(
        self,
        activation_BMD: t.Tensor,
    ) -> ForwardResult:
        """returns the activations, the h states, and the reconstructed activations"""
        self._validate_acts_shape(activation_BMD)

        latents_BH = self._encode(activation_BMD)

        latents_shared_BHs = latents_BH[:, : self.n_shared_latents]
        latents_indep_BHi = latents_BH[:, self.n_shared_latents :]

        if self._W_dec_shared_m0_HsD is not self._W_dec_shared_m1_HsD:
            logger.warning("Training without tied decoder weights!")

        recon_BMD = self._decode(latents_shared_BHs, latents_indep_BHi)

        res = self.ForwardResult(
            hidden_shared_BHs=latents_shared_BHs,
            hidden_indep_BHi=latents_indep_BHi,
            recon_acts_BMD=recon_BMD,
        )

        if res.recon_acts_BMD.shape != activation_BMD.shape:
            raise ValueError(
                f"recon_acts_BMD.shape {res.recon_acts_BMD.shape} != activation_BMD.shape {activation_BMD.shape}"
            )

        return res

    def _join_shared_indep(self, shared: t.Tensor, indep: t.Tensor, dim: int) -> t.Tensor:
        assert shared.shape[dim] == self.n_shared_latents
        assert indep.shape[dim] == self.n_latents_total - self.n_shared_latents
        return t.cat([shared, indep], dim=dim)

    def _decode(self, latents_shared_BHs: t.Tensor, latents_indep_BHi: t.Tensor) -> t.Tensor:
        # decoder for shared latents
        pre_bias_shared_m0_BD = latents_shared_BHs @ self._W_dec_shared_m0_HsD
        pre_bias_shared_m1_BD = latents_shared_BHs @ self._W_dec_shared_m1_HsD
        pre_bias_shared_BMD = t.stack([pre_bias_shared_m0_BD, pre_bias_shared_m1_BD], dim=1)

        # decoder for independent latents
        pre_bias_indep_BMD = einsum(latents_indep_BHi, self._W_dec_indep_HiMD, "b h_indep, h_indep m d -> b m d")

        # combine the two
        pre_bias_BMD = pre_bias_shared_BMD + pre_bias_indep_BMD

        # add bias
        return pre_bias_BMD + self.b_dec_MD

    def _encode(self, activation_BMD: t.Tensor) -> t.Tensor:
        pre_bias_BH = einsum(
            activation_BMD,
            self.W_enc_MDH,
            "b ..., ... h -> b h",
        )
        pre_activation_BH = pre_bias_BH + self.b_enc_H
        return self.activation_fn.forward(pre_activation_BH)

    def forward(self, activation_BMD: t.Tensor) -> t.Tensor:
        self._validate_acts_shape(activation_BMD)
        latents_BH = self._encode(activation_BMD)
        latents_shared_BHs = latents_BH[:, : self.n_shared_latents]
        latents_indep_BHi = latents_BH[:, self.n_shared_latents :]
        return self._decode(latents_shared_BHs, latents_indep_BHi)

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_latents_total": self.n_latents_total,
            "n_shared_latents": self.n_shared_latents,
            "activation_fn_classname": self.activation_fn.__class__.__name__,
            "activation_fn_cfg": self.activation_fn._dump_cfg(),
            # don't need to serialize init_strategy as loading from state_dict will re-initialize the params
        }

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "DiffingCrosscoder[TActivation]":
        d_model = runtimecast(cfg["d_model"], int)
        n_latents_total = runtimecast(cfg["n_latents_total"], int)
        n_shared_latents = runtimecast(cfg["n_shared_latents"], int)

        activation_fn_cls = ACTIVATIONS_MAP.get(runtimecast(cfg["activation_fn_classname"], str))
        if activation_fn_cls is None:
            raise ValueError(f"Activation function class {cfg['activation_fn_classname']} not found")

        activation_fn = cast(TActivation, activation_fn_cls._from_cfg(cfg["activation_fn_cfg"]))

        return DiffingCrosscoder(
            d_model=d_model,
            n_latents_total=n_latents_total,
            n_shared_latents=n_shared_latents,
            activation_fn=activation_fn,
            # don't need to serialize init_strategy as loading from state_dict will re-initialize the params
        )

    def _validate_acts_shape(self, activation_BMD: t.Tensor) -> None:
        n_models, d_model = activation_BMD.shape[1:]
        if d_model != self.d_model:
            raise ValueError(f"d_model {d_model} != self.d_model {self.d_model}")
        if n_models != N_MODELS:
            raise ValueError(f"n_models {n_models} != N_MODELS {N_MODELS}")

    def with_decoder_unit_norm(self) -> "DiffingCrosscoder[TActivation]":
        """
        returns a copy of the model with the weights rescaled such that the decoder norm of each feature is one,
        but the model makes the same predictions.
        """

        cc = DiffingCrosscoder(
            d_model=self.d_model,
            n_latents_total=self.n_latents_total,
            n_shared_latents=self.n_shared_latents,
            activation_fn=self.activation_fn,
        )

        cc.load_state_dict(self.state_dict())

        cc.make_decoder_max_unit_norm_()

        return cc

    @t.no_grad()
    def make_decoder_max_unit_norm_(self) -> None:
        output_space_norms_HM = l2_norm(self.theoretical_decoder_W_dec_HMD(), dim=-1)
        max_norms_per_latent_H = output_space_norms_HM.amax(dim=1)

        self.W_enc_MDH.mul_(max_norms_per_latent_H)
        self.b_enc_H.mul_(max_norms_per_latent_H)

        max_norms_per_latent_shared_Hs = max_norms_per_latent_H[: self.n_shared_latents]
        max_norms_per_latent_indep_Hi = max_norms_per_latent_H[self.n_shared_latents :]

        self._W_dec_indep_HiMD.div_(max_norms_per_latent_indep_Hi[..., None, None])

        # IMPORTANT: here we untie _W_dec_indep_HiMD and _W_dec_shared_m0_HsD to divide by seperate scaling factors
        self._W_dec_shared_m0_HsD = nn.Parameter(self._W_dec_shared_m0_HsD / max_norms_per_latent_shared_Hs[..., None])
        self._W_dec_shared_m1_HsD = nn.Parameter(self._W_dec_shared_m1_HsD / max_norms_per_latent_shared_Hs[..., None])

    def with_activation_scaling(self, scaling_factors_M: t.Tensor) -> "DiffingCrosscoder[TActivation]":
        cc = DiffingCrosscoder(
            d_model=self.d_model,
            n_latents_total=self.n_latents_total,
            n_shared_latents=self.n_shared_latents,
            activation_fn=self.activation_fn,
        )

        cc.load_state_dict(self.state_dict())

        cc.fold_activation_scaling_into_weights_(scaling_factors_M)

        return cc


    @contextmanager
    def temporarily_fold_activation_scaling(self, scaling_factors_M: t.Tensor):
        """Temporarily fold scaling factors into weights."""
        self.fold_activation_scaling_into_weights_(scaling_factors_M)
        yield
        _ = self.unfold_activation_scaling_from_weights_()

    @t.no_grad()
    def fold_activation_scaling_into_weights_(self, scaling_factors_M: t.Tensor) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the m can be run on raw llm activations."""
        if self.is_folded.item():
            raise ValueError("Scaling factors already folded into weights")

        if scaling_factors_M.shape != (N_MODELS,):
            raise ValueError(f"Expected shape ({N_MODELS},), got {scaling_factors_M.shape}")
        if (scaling_factors_M == 0).any():
            raise ValueError("Scaling factors contain zeros, which would lead to division by zero")
        if scaling_factors_M.isnan().any() or scaling_factors_M.isinf().any():
            raise ValueError("Scaling factors contain NaN or Inf values")

        scaling_factors_M = scaling_factors_M.to(self.W_enc_MDH.device)

        self.W_enc_MDH.mul_(scaling_factors_M[..., None, None])

        self._W_dec_indep_HiMD.div_(scaling_factors_M[..., None])

        # IMPORTANT. Here we untie _W_dec_shared_m1_HsD and _W_dec_shared_m2_HsD to divide by seperate scaling factors
        self._W_dec_shared_m0_HsD = nn.Parameter(self._W_dec_shared_m0_HsD / scaling_factors_M[0])
        self._W_dec_shared_m1_HsD = nn.Parameter(self._W_dec_shared_m1_HsD / scaling_factors_M[1])

        self.b_dec_MD.div_(scaling_factors_M[..., None])

        # set buffer to prevent double-folding
        self.folded_scaling_factors_M = scaling_factors_M
        self.is_folded = t.tensor(True, dtype=t.bool)

    @t.no_grad()
    def unfold_activation_scaling_from_weights_(self) -> t.Tensor:
        if not self.is_folded.item():
            raise ValueError("No folded scaling factors found")
        if self.folded_scaling_factors_M is None:
            raise ValueError("Inconsistent state: is_folded is True but folded_scaling_factors_M is None")

        folded_scaling_factors_M = self.folded_scaling_factors_M.clone()

        self.W_enc_MDH.div_(folded_scaling_factors_M[..., None, None])

        self._W_dec_indep_HiMD.mul_(folded_scaling_factors_M[..., None])

        W_dec_shared_m0_HsD = self._W_dec_shared_m0_HsD * folded_scaling_factors_M[0]
        W_dec_shared_m1_HsD = self._W_dec_shared_m1_HsD * folded_scaling_factors_M[1]
        assert t.allclose(W_dec_shared_m0_HsD, W_dec_shared_m1_HsD), (
            "expected unscaled shared decoder weights to be equal"
        )

        # IMPORTANT. Here we retie _W_dec_shared_m1_HsD and _W_dec_shared_m2_HsD
        self._W_dec_shared_m0_HsD = nn.Parameter(W_dec_shared_m0_HsD)
        self._W_dec_shared_m1_HsD = self._W_dec_shared_m0_HsD

        self.b_dec_MD.mul_(folded_scaling_factors_M[..., None])

        self.folded_scaling_factors_M = None
        self.is_folded = t.tensor(False, dtype=t.bool)

        return folded_scaling_factors_M
