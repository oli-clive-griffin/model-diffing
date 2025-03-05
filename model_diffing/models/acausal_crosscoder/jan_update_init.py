from math import prod

import torch
from einops import rearrange

from model_diffing.models import InitStrategy
from model_diffing.models.acausal_crosscoder.acausal_crosscoder import AcausalCrosscoder
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.models.utils.jan_update_init import BaseDataDependentJumpReLUInitStrategy


class DataDependentJumpReLUInitStrategy(
    BaseDataDependentJumpReLUInitStrategy,
    InitStrategy[AcausalCrosscoder[AnthropicJumpReLUActivation]],
):
    """
    Implementation of the initialization scheme described in:
        https://transformer-circuits.pub/2025/january-update/index.html.
    """

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[AnthropicJumpReLUActivation]) -> None:
        n = prod(cc.crosscoding_dims) * cc.d_model
        m = cc.hidden_dim

        cc.W_dec_HXD.uniform_(-1.0 / n, 1.0 / n)
        cc.W_enc_XDH.copy_(
            rearrange(cc.W_dec_HXD, "hidden ... -> ... hidden")  #
            * (n / m)
        )

        cc.b_enc_H.copy_(self.get_calibrated_b_enc_H(cc.W_enc_XDH, cc.hidden_activation))
        cc.b_dec_XD.zero_()
