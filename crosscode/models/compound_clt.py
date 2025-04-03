from dataclasses import dataclass
from typing import Generic, cast

import torch
from torch import nn

from crosscode.models.base_crosscoder import TActivation
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder


class CompoundCrossLayerTranscoder(Generic[TActivation]):
    def __init__(
        self,
        n_hookpoints: int,
        d_model: int,
        n_latents_per_tc: int,
        activation_fn: TActivation,
        linear_skip: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_hookpoints = n_hookpoints
        self.n_hookpoints_out = self.n_hookpoints - 1
        self.transcoders = nn.ModuleList(
            [
                CrossLayerTranscoder(
                    d_model=d_model,
                    n_layers_out=self.n_hookpoints_out - i,
                    n_latents=n_latents_per_tc,
                    activation_fn=activation_fn,
                    use_encoder_bias=False,
                    use_decoder_bias=False,
                    linear_skip=linear_skip,
                    dtype=dtype,
                )
                for i in range(self.n_hookpoints - 1)
            ]
        )
        self.n_hookpoints = n_hookpoints
        self._dtype = dtype
        self.n_latents_per_tc = n_latents_per_tc

    @dataclass
    class ForwardResult:
        output_BPoD: torch.Tensor
        latents_BPoL: torch.Tensor
        pre_activations_BPoL: torch.Tensor

    def forward(self, activation_BPD: torch.Tensor) -> ForwardResult:
        # P = all hookpoints has length self.n_hookpoints
        # Po = out hookpoints = self.n_hookpoints - 1
        B, P, D = activation_BPD.shape
        Po = P - 1
        device = activation_BPD.device
        dtype = activation_BPD.dtype
        output_BPoD = torch.zeros((B, Po, D), dtype=dtype, device=device)
        latents_BPoL = torch.zeros((B, Po, self.n_latents_per_tc), dtype=dtype, device=device)
        pre_activations_BPoL = torch.zeros((B, Po, self.n_latents_per_tc), dtype=dtype, device=device)

        for input_idx in range(self.n_hookpoints - 1):
            transcoder = cast(CrossLayerTranscoder[TActivation], self.transcoders[input_idx])
            input_BD = activation_BPD[:, input_idx]  # we should never take the last hookpoint as input
            train_output = transcoder.forward_train(input_BD)

            # output is different to latents and pre_activations, as it's cumulatively built up, with transcoder i
            # outputting from hookpoint i+1 onwards
            output_BPoD[:, input_idx:, :] += train_output.output_BPD

            latents_BPoL[:, input_idx, :] = train_output.latents_BL
            pre_activations_BPoL[:, input_idx, :] = train_output.pre_activations_BL

        return self.ForwardResult(
            output_BPoD=output_BPoD,
            latents_BPoL=latents_BPoL,
            pre_activations_BPoL=pre_activations_BPoL,
        )
