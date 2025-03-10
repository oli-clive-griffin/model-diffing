import torch
from einops import rearrange

from model_diffing.models import InitStrategy
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.models.diffing_crosscoder.diffing_crosscoder import N_MODELS, DiffingCrosscoder
from model_diffing.models.utils.jan_update_init import BaseDataDependentJumpReLUInitStrategy


class ModelDiffingDataDependentJumpReLUInitStrategy(
    BaseDataDependentJumpReLUInitStrategy, InitStrategy[DiffingCrosscoder[AnthropicJumpReLUActivation]]
):
    """
    Implementation of the initialization scheme described in:
        https://transformer-circuits.pub/2025/january-update/index.html.

    Adapted for the model-diffing crosscoder case described in:
        https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html
    """

    @torch.no_grad()
    def init_weights(self, cc: DiffingCrosscoder[AnthropicJumpReLUActivation]) -> None:
        n = cc.d_model * N_MODELS
        m = cc.hidden_dim

        # initialise W_dec from U(-1/n, 1/n)
        cc._W_dec_shared_m0_HsD.uniform_(-1.0 / n, 1.0 / n)
        cc._W_dec_indep_HiMD.uniform_(-1.0 / n, 1.0 / n)

        cc.W_enc_MDH.copy_(
            rearrange(cc.theoretical_decoder_W_dec_HMD(), "hidden ... -> ... hidden")  #
            * (n / m)
        )

        cc.b_enc_H.copy_(self.get_calibrated_b_enc_H(cc.W_enc_MDH, cc.activation_fn).to(cc.b_enc_H.device))
        cc.b_dec_MD.zero_()
