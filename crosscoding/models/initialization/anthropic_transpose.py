from typing import Any

import torch
from einops import rearrange, repeat

from crosscoding.models.initialization.init_strategy import InitStrategy
from crosscoding.models.sparse_coders import CrossLayerTranscoder, ModelHookpointAcausalCrosscoder
from crosscoding.utils import random_direction_init_


class AnthropicTransposeInit(InitStrategy[ModelHookpointAcausalCrosscoder[Any]]):
    def __init__(self, dec_init_norm: float):
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: ModelHookpointAcausalCrosscoder[Any]) -> None:
        random_direction_init_(cc.W_dec_LMPD, self.dec_init_norm)

        cc.W_enc_MPDL.copy_(rearrange(cc.W_dec_LMPD.clone(), "l ... -> ... l"))

        if cc.b_enc_L is not None:
            cc.b_enc_L.zero_()
        if cc.b_dec_MPD is not None:
            cc.b_dec_MPD.zero_()


class AnthropicTransposeInitCrossLayerTC(InitStrategy[CrossLayerTranscoder[Any]]):
    def __init__(self, dec_init_norm: float):
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: CrossLayerTranscoder[Any]) -> None:
        """doing this backwards because we have multiple out layers, only one in layer. This should be symmmetrical though so it's fine?"""
        random_direction_init_(cc.W_enc_DL, self.dec_init_norm)

        W_dec_LD = rearrange(cc.W_enc_DL.clone(), "d l -> l d")
        W_dec_LPD = repeat(W_dec_LD, "l d -> l p d", p=cc.n_layers_out)
        cc.W_dec_LPD.copy_(W_dec_LPD)

        if cc.b_enc_L is not None:
            cc.b_enc_L.zero_()
        if cc.b_dec_PD is not None:
            cc.b_dec_PD.zero_()
