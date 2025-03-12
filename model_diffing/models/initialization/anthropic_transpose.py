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
        random_direction_init_(cc.W_dec_HXD, self.dec_init_norm)

        cc.W_enc_XDH.copy_(rearrange(cc.W_dec_HXD.clone(), "h ... -> ... h"))

        cc.b_enc_H.zero_()
        cc.b_dec_XD.zero_()

