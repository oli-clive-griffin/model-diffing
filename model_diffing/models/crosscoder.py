from dataclasses import dataclass

import torch as t
from einops import einsum, rearrange
from torch import nn

"""
Dimensions:
- M: number of models
- B: batch size
- L: number of layers
- D: Subject model dimension
- H: Autoencoder hidden dimension
"""

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
            W_dec_norm_MLD1 = self.W_dec_HMLD.norm(dim=-1, keepdim=True)
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

    def forward(self, activation_BMLD: t.Tensor) -> t.Tensor:
        assert activation_BMLD.shape[1:] == self.activations_shape
        hidden_BH = self.encode(activation_BMLD)
        assert hidden_BH.shape[1:] == (self.hidden_dim,)
        reconstructed_BMLD = self.decode(hidden_BH)
        assert reconstructed_BMLD.shape[1:] == self.activations_shape
        return reconstructed_BMLD

    def sparsity_loss(self, hidden_BH: t.Tensor) -> t.Tensor:
        assert (hidden_BH >= 0).all()
        # separate norms for each (model, layer)
        W_dec_l2_norms_HML = self.W_dec_HMLD.norm(dim=-1, p=2)
        # a norm's weight is the sum of it's (model, layer) decoder vector norm
        summed_norms_H = W_dec_l2_norms_HML.sum(dim=(1, 2))
        weighted_hidden_BH = hidden_BH * summed_norms_H
        l1_hidden_B = weighted_hidden_BH.norm(p=1, dim=1)
        return l1_hidden_B.mean()

    def forward_train(self, activation_BLD: t.Tensor) -> tuple[t.Tensor, Losses]:
        hidden_BH = self.encode(activation_BLD)
        reconstructed_BLD = self.decode(hidden_BH)

        losses = Losses(
            reconstruction_loss=reconstruction_loss(reconstructed_BLD, activation_BLD),
            sparsity_loss=self.sparsity_loss(hidden_BH),
        )

        return reconstructed_BLD, losses


def reconstruction_loss(activation_BLD: t.Tensor, target_BLD: t.Tensor) -> t.Tensor:
    x_NL = (activation_BLD - target_BLD).norm(dim=-1).square()
    x_N = x_NL.sum(dim=-1)
    return x_N.mean()
