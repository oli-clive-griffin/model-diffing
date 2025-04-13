import torch
from torch import nn

from crosscode.models.activations.topk import BatchTopkActivation

# d_model -> n_latents -> d_mlp -> n_latents -> d_model


class SparseFeatureCoder(nn.Module):
    """replaces an MLP"""

    def __init__(self, d_model: int, n_latents: int, d_mlp: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.d_mlp = d_mlp

        # for encoder and decoder
        self.sparse_fn = BatchTopkActivation(k_per_example=k)

        # encoder
        self.W_enc_sparse_DL = nn.Parameter(torch.empty((d_model, n_latents)))

        # MLP
        self.W_enc_mlp_LM = nn.Parameter(torch.empty((n_latents, d_mlp)))
        self.b_enc_mlp_M = nn.Parameter(torch.empty((d_mlp,)))
        self.W_dec_mlp_ML = nn.Parameter(torch.empty((d_mlp, n_latents)))
        self.b_dec_mlp_L = nn.Parameter(torch.empty((n_latents,)))

        # decoder
        self.W_dec_sparse_LD = nn.Parameter(torch.empty((n_latents, d_model)))

        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.W_enc_sparse_DL)
            torch.nn.init.kaiming_normal_(self.W_enc_mlp_LM)
            torch.nn.init.kaiming_normal_(self.W_dec_mlp_ML)
            torch.nn.init.kaiming_normal_(self.W_dec_sparse_LD)

    def encode_sparse_DL(self, acts_in_BD: torch.Tensor) -> torch.Tensor:
        return self.sparse_fn(acts_in_BD @ self.W_enc_sparse_DL)

    def mlp_forward_BL(self, latents_in_BL: torch.Tensor) -> torch.Tensor:
        mlp_hidden_BM = torch.relu(latents_in_BL @ self.W_enc_mlp_LM + self.b_enc_mlp_M)
        return mlp_hidden_BM @ self.W_dec_mlp_ML + self.b_dec_mlp_L

    def decode_from_sparse_BD(self, latents_out_BL: torch.Tensor) -> torch.Tensor:
        return latents_out_BL @ self.W_dec_sparse_LD

    def forward(self, acts_in_BD: torch.Tensor) -> torch.Tensor:
        latents_in_BL = self.encode_sparse_DL(acts_in_BD)
        latents_out_BL = self.sparse_fn(self.mlp_forward_BL(latents_in_BL))
        return self.decode_from_sparse_BD(latents_out_BL)

