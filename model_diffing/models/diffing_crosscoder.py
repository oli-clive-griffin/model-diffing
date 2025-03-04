from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import torch as t
from einops import einsum, repeat
from torch import nn

from model_diffing.log import logger
from model_diffing.models.acausal_crosscoder import InitStrategy
from model_diffing.models.activations import ACTIVATIONS_MAP
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.utils import SaveableModule

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

    # is_folded: t.Tensor
    # folded_scaling_factors_M: t.Tensor | None

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_explicitly_shared_latents: int,
        hidden_activation: TActivation,
        init_strategy: InitStrategy["DiffingCrosscoder[TActivation]"] | None = None,
    ):
        super().__init__()

        self.d_model = d_model

        assert n_explicitly_shared_latents <= hidden_dim
        if n_explicitly_shared_latents > hidden_dim / 10:
            logger.warning(
                f"n_explicitly_shared_latents {n_explicitly_shared_latents} is greater than 10% of hidden_dim {hidden_dim}, are you sure this was intentional?"
            )

        self.hidden_dim = hidden_dim
        self.n_shared_latents = n_explicitly_shared_latents

        self.W_enc_MDH = nn.Parameter(t.empty((N_MODELS, d_model, hidden_dim)))
        self.b_enc_H = nn.Parameter(t.empty((hidden_dim,)))

        self.hidden_activation = hidden_activation

        self._W_dec_shared_HsD = nn.Parameter(t.empty((n_explicitly_shared_latents, d_model)))
        """decoder weights for the shared features"""

        self._W_dec_indep_HiMD = nn.Parameter(t.empty((hidden_dim - n_explicitly_shared_latents, N_MODELS, d_model)))
        """decoder weights for the model-specific features"""

        self.b_dec_MD = nn.Parameter(t.empty((N_MODELS, d_model)))

        if init_strategy is not None:
            init_strategy.init_weights(self)

        # # Initialize the buffer with a zero tensor of the correct shape, this means it's always serialized
        # self.register_buffer("folded_scaling_factors_M", t.zeros(N_MODELS))
        # # However, track whether it's actually holding a meaningful value by using this boolean flag.
        # # Represented as a tensor so that it's serialized by torch.save
        # self.register_buffer("is_folded", t.tensor(False, dtype=t.bool))

    def theoretical_decoder_W_dec_HMD(self) -> t.Tensor:
        """
        This is not a real weight matrix, it's theoretically what the decoder WOULD be without the weight sharing
        """
        W_dec_shared_HsMD = repeat(self._W_dec_shared_HsD, "h_shared d -> h_shared model d", model=2)

        # IMPORTANT: shared latents are the FIRST `n_shared_latents` along the hidden dim, ordering matters
        W_dec_HMD = t.cat([W_dec_shared_HsMD, self._W_dec_indep_HiMD], dim=0)

        return W_dec_HMD

    @dataclass
    class ForwardResult:
        hidden_BHs: t.Tensor
        hidden_BHi: t.Tensor
        recon_acts_BMD: t.Tensor

    def forward_train(
        self,
        activation_BMD: t.Tensor,
    ) -> ForwardResult:
        """returns the activations, the h states, and the reconstructed activations"""
        self._validate_acts_shape(activation_BMD)

        latents_shared_BHs, latents_indep_BHi = self._encode(activation_BMD)

        recon_BMD = self._decode(latents_shared_BHs, latents_indep_BHi)

        res = self.ForwardResult(
            hidden_BHs=latents_shared_BHs,
            hidden_BHi=latents_indep_BHi,
            recon_acts_BMD=recon_BMD,
        )

        if res.recon_acts_BMD.shape != activation_BMD.shape:
            raise ValueError(
                f"recon_acts_BMD.shape {res.recon_acts_BMD.shape} != activation_BMD.shape {activation_BMD.shape}"
            )

        return res

    def _decode(self, latents_shared_BHs: t.Tensor, latents_indep_BHi: t.Tensor) -> t.Tensor:
        # decoder for shared latents
        pre_bias_shared_BD = einsum(latents_shared_BHs, self._W_dec_shared_HsD, "b h_shared, h_shared d -> b d")
        pre_bias_shared_BMD = repeat(pre_bias_shared_BD, "b d -> b model d", model=N_MODELS)

        # decoder for independent latents
        pre_bias_indep_BMD = einsum(latents_indep_BHi, self._W_dec_indep_HiMD, "b h_indep, h_indep m d -> b m d")

        pre_bias_BMD = pre_bias_shared_BMD + pre_bias_indep_BMD

        return pre_bias_BMD + self.b_dec_MD

    def _encode(self, activation_BMD: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        pre_bias_BH = einsum(
            activation_BMD,
            self.W_enc_MDH,
            "b ..., ... h -> b h",
        )
        pre_activation_BH = pre_bias_BH + self.b_enc_H
        latents_BH: t.Tensor = self.hidden_activation.forward(pre_activation_BH)
        latents_shared_BHs = latents_BH[:, : self.n_shared_latents]
        latents_indep_BHi = latents_BH[:, self.n_shared_latents :]
        return latents_shared_BHs, latents_indep_BHi

    def forward(self, activation_BMD: t.Tensor) -> t.Tensor:
        return self.forward_train(activation_BMD).recon_acts_BMD

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "n_shared_latents": self.n_shared_latents,
            "hidden_activation_classname": self.hidden_activation.__class__.__name__,
            "hidden_activation_cfg": self.hidden_activation._dump_cfg(),
            # don't need to serialize init_strategy as loading from state_dict will re-initialize the params
        }

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "DiffingCrosscoder[TActivation]":
        hidden_activation_cfg = cfg["hidden_activation_cfg"]
        hidden_activation_classname = cfg["hidden_activation_classname"]

        hidden_activation_cls = ACTIVATIONS_MAP[hidden_activation_classname]
        hidden_activation = cast(TActivation, hidden_activation_cls._from_cfg(hidden_activation_cfg))

        return DiffingCrosscoder(
            d_model=cfg["d_model"],
            hidden_dim=cfg["hidden_dim"],
            n_explicitly_shared_latents=cfg["n_explicitly_shared_latents"],
            hidden_activation=hidden_activation,
            # don't need to serialize init_strategy as loading from state_dict will re-initialize the params
        )

    # def with_decoder_unit_norm(self) -> "DiffingCrosscoder[TActivation]":
    #     """
    #     returns a copy of the model with the weights rescaled such that the decoder norm of each feature is one,
    #     but the model makes the same predictions.
    #     """

    #     raise NotImplementedError("Not implemented")

    #     cc = DiffingCrosscoder(
    #         d_model=self.d_model,
    #         hidden_dim=self.hidden_dim,
    #         hidden_activation=self.hidden_activation,
    #     )

    #     cc.make_decoder_max_unit_norm_()

    #     return cc

    # def make_decoder_max_unit_norm_(self) -> None:
    #     """
    #     scales the decoder weights such that the model makes the same predictions, but for
    #     each latent, the maximum norm of it's decoder vectors is 1.

    #     For example, in a 2-model, 3-hookpoint crosscoder, the norms for a given latent might be scaled to:

    #     [[1, 0.2],
    #      [0.2, 0.4],
    #      [0.1, 0.3]]
    #     """
    #     raise NotImplementedError("Not implemented")
    #     with t.no_grad():
    #         output_space_norms_HM = reduce(self.W_dec_HMD, "h m d -> h m", l2_norm)
    #         max_norms_per_latent_H = output_space_norms_HM.amax(dim=1)  # all but the first dimension
    #         assert max_norms_per_latent_H.shape == (self.hidden_dim,)

    #         # this means that the maximum norm of the decoder vectors into a given output space is 1
    #         # for example, in a cross-model cc, the norms for each model might be (1, 0.2) or (0.2, 1) or (1, 1)
    #         self.W_dec_HMD.copy_(self.W_dec_HMD / max_norms_per_latent_H[..., None, None])
    #         self.W_enc_MDH.copy_(self.W_enc_MDH * max_norms_per_latent_H)
    #         self.b_enc_H.copy_(self.b_enc_H * max_norms_per_latent_H)
    #         # no alteration needed for self.b_dec_MD

    # @t.no_grad()
    # def _scale_weights(self, scaling_factors_2: t.Tensor) -> None:
    #     raise NotImplementedError("Not implemented")

    #     self.W_enc_MDH.mul_(scaling_factors_2[..., None, None])
    #     self.W_dec_HMD.div_(scaling_factors_2[..., None])
    #     self.b_dec_MD.div_(scaling_factors_2[..., None])

    def _validate_scaling_factors(self, scaling_factors_M: t.Tensor) -> None:
        if t.any(scaling_factors_M == 0):
            raise ValueError("Scaling factors contain zeros, which would lead to division by zero")
        if t.any(t.isnan(scaling_factors_M)) or t.any(t.isinf(scaling_factors_M)):
            raise ValueError("Scaling factors contain NaN or Inf values")

    # @contextmanager
    # def temporarily_fold_activation_scaling(self, scaling_factors_M: t.Tensor):
    #     """Temporarily fold scaling factors into weights."""
    #     self.fold_activation_scaling_into_weights_(scaling_factors_M)
    #     yield
    #     _ = self.unfold_activation_scaling_from_weights_()

    # def fold_activation_scaling_into_weights_(self, scaling_factors_M: t.Tensor) -> None:
    #     """scales the crosscoder weights by the activation scaling factors, so that the m can be run on raw llm activations."""
    #     if self.is_folded.item():
    #         raise ValueError("Scaling factors already folded into weights")

    #     self._validate_scaling_factors(scaling_factors_M)
    #     scaling_factors_M = scaling_factors_M.to(self.W_enc_MDH.device)
    #     self._scale_weights(scaling_factors_M)

    #     # set buffer to prevent double-folding
    #     self.folded_scaling_factors_M = scaling_factors_M
    #     self.is_folded = t.tensor(True, dtype=t.bool)

    # def unfold_activation_scaling_from_weights_(self) -> t.Tensor:
    #     if not self.is_folded.item():
    #         raise ValueError("No folded scaling factors found")

    #     if self.folded_scaling_factors_M is None:
    #         raise ValueError("Inconsistent state: is_folded is True but folded_scaling_factors_M is None")

    #     folded_scaling_factors_M = self.folded_scaling_factors_M.clone()
    #     # Clear the buffer before operations to prevent double-unfolding

    #     self.folded_scaling_factors_M = None
    #     self.is_folded = t.tensor(False, dtype=t.bool)

    #     self._scale_weights(1 / folded_scaling_factors_M)

    #     return folded_scaling_factors_M

    def _validate_acts_shape(self, activation_BMD: t.Tensor) -> None:
        n_models, d_model = activation_BMD.shape[1:]
        if d_model != self.d_model:
            raise ValueError(f"d_model {d_model} != self.d_model {self.d_model}")
        if n_models != N_MODELS:
            raise ValueError(f"n_models {n_models} != N_MODELS {N_MODELS}")


if __name__ == "__main__":
    from model_diffing.models.activations import ReLUActivation

    class RandomInit(InitStrategy[DiffingCrosscoder[ActivationFunction]]):
        @t.no_grad()
        def init_weights(self, cc: DiffingCrosscoder[ActivationFunction]) -> None:
            cc.W_enc_MDH.random_()
            cc.b_enc_H.random_()

            cc._W_dec_indep_HiMD.random_()
            cc._W_dec_shared_HsD.random_()
            cc.b_dec_MD.random_()

    n_latents = 4
    shared_latents = 1

    d_model = 5

    cc = DiffingCrosscoder(  # type: ignore
        d_model=d_model,
        hidden_dim=n_latents,
        n_explicitly_shared_latents=shared_latents,
        hidden_activation=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    activation_BMD = t.randn(1, N_MODELS, d_model)
    res = cc.forward_train(activation_BMD)
    mse = (res.recon_acts_BMD - activation_BMD).pow(2).mean()
    mse.backward()

    print(f"{cc._W_dec_shared_HsD.grad=}")
    print(f"{cc._W_dec_indep_HiMD.grad=}")

    assert cc._W_dec_shared_HsD.grad is not None
    assert cc._W_dec_indep_HiMD.grad is not None

    theoretical_grad_W_dec_HMD = t.cat(
        [
            repeat(cc._W_dec_shared_HsD.grad, "h_shared d -> h_shared model d", model=2),
            cc._W_dec_indep_HiMD.grad,
        ],
        dim=0,
    )

    print(f"{theoretical_grad_W_dec_HMD=}")
