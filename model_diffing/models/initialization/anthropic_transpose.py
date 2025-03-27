from typing import Any

import torch
from einops import rearrange

from model_diffing.models.acausal_crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.utils import random_direction_init_


class AnthropicTransposeInit(InitStrategy[AcausalCrosscoder[Any]]):
    def __init__(self, dec_init_norm: float):
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[Any]) -> None:
        random_direction_init_(cc.W_dec_LXD, self.dec_init_norm)

        cc.W_enc_XDL.copy_(rearrange(cc.W_dec_LXD.clone(), "h ... -> ... h"))

        if cc.b_enc_L is not None:
            cc.b_enc_L.zero_()
        if cc.b_dec_XD is not None:
            cc.b_dec_XD.zero_()
