from dataclasses import dataclass

import torch as t
from einops import einsum, rearrange, reduce
from torch import nn

from model_diffing.utils import l2_norm

"""
Dimensions:
- M: number of models
- B: batch size
- L: number of layers
- D: Subject model dimension
- H: Autoencoder hidden dimension
"""

# hacky but useful for debugging
t.Tensor.d = lambda self: f"{self.shape}, dtype={self.dtype}, device={self.device}"  # type: ignore


@dataclass
class Losses:
    reconstruction_loss: t.Tensor
    sparsity_loss: t.Tensor


class AcausalCrosscoder(nn.Module):
    """crosscoder that autoencodes activations of a subset of a model's layers"""

    def __init__(
        self,
        n_models: int,
        n_layers: int,
        d_model: int,
        hidden_dim: int,
        dec_init_norm: float,
    ):
        super().__init__()
        self.activations_shape = (n_models, n_layers, d_model)
        self.hidden_dim = hidden_dim

        self.W_enc_MLDH = nn.Parameter(t.randn((n_models, n_layers, d_model, hidden_dim)))
        self.b_enc_H = nn.Parameter(t.zeros((hidden_dim,)))

        self.W_dec_HMLD = nn.Parameter(
            rearrange(  # "transpose" of the encoder weights
                self.W_enc_MLDH.clone(),
                "model layer d_model hidden -> hidden model layer d_model",
            )
        )
        self.b_dec_MLD = nn.Parameter(t.zeros((n_models, n_layers, d_model)))

        with t.no_grad():
            W_dec_norm_MLD1 = l2_norm(self.W_dec_HMLD, dim=-1, keepdim=True)
            self.W_dec_HMLD.div_(W_dec_norm_MLD1)
            self.W_dec_HMLD.mul_(dec_init_norm)

    def encode(self, activation_BMLD: t.Tensor) -> t.Tensor:
        hidden_BH = einsum(
            activation_BMLD,
            self.W_enc_MLDH,
            "batch model layer d_model, model layer d_model hidden -> batch hidden",
        )
        hidden_BH = hidden_BH + self.b_enc_H
        return t.relu(hidden_BH)

    def decode(self, hidden_BH: t.Tensor) -> t.Tensor:
        activation_BMLD = einsum(
            hidden_BH,
            self.W_dec_HMLD,
            "batch hidden, hidden model layer d_model -> batch model layer d_model",
        )
        activation_BMLD += self.b_dec_MLD
        return activation_BMLD

    def reconstruction_loss(self, activation_BMLD: t.Tensor, target_BMLD: t.Tensor) -> t.Tensor:
        """This is a little weird because we have both model and layer dimensions, so it's worth explaining deeply:

        The reconstruction loss is a sum of squared L2 norms of the error for each activation space being reconstructed.
        In the Anthropic crosscoders update, they don't write for the multiple-model case, so they write it as:

        $$ \sum_{l \in L} \|a^l(x_j) - a^{l'}(x_j)\|^2 $$

        Here, I'm assuming we want to expand that sum to be over models, so we would have:

        $$ \sum_{m \in M} \sum_{l \in L} \|a_m^l(x_j) - a_m^{l'}(x_j)\|^2 $$
        """
        error_BMLD = activation_BMLD - target_BMLD
        error_norm_BML = reduce(error_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
        squared_error_norm_BML = error_norm_BML.square()
        summed_squared_error_norm_B = reduce(squared_error_norm_BML, "batch model layer -> batch", t.sum)
        return summed_squared_error_norm_B.mean()

    def sparsity_loss(self, hidden_BH: t.Tensor) -> t.Tensor:
        assert (hidden_BH >= 0).all()
        # each latent has a separate norms for each (model, layer)
        W_dec_l2_norms_HML = reduce(self.W_dec_HMLD, "hidden model layer dim -> hidden model layer", l2_norm)
        # to get the weighting factor for each latent, we sum it's decoder norms for each (model, layer)
        summed_norms_H = reduce(W_dec_l2_norms_HML, "hidden model layer -> hidden", t.sum)
        # now we weight the latents by the sum of their norms
        weighted_hidden_BH = hidden_BH * summed_norms_H
        summed_weighted_hidden_B = reduce(weighted_hidden_BH, "batch hidden -> batch", t.sum)
        return summed_weighted_hidden_B.mean()

    def forward_train(self, activation_BMLD: t.Tensor) -> tuple[t.Tensor, Losses]:
        hidden_BH = self.encode(activation_BMLD)
        reconstructed_BMLD = self.decode(hidden_BH)
        assert reconstructed_BMLD.shape == activation_BMLD.shape
        assert len(reconstructed_BMLD.shape) == 4

        losses = Losses(
            reconstruction_loss=self.reconstruction_loss(reconstructed_BMLD, activation_BMLD),
            sparsity_loss=self.sparsity_loss(hidden_BH),
        )

        return reconstructed_BMLD, losses
