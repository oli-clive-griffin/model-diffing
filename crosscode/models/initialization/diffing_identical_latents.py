from typing import Any

import torch

from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
from crosscode.models.initialization.init_strategy import InitStrategy


class IdenticalLatentsInit(InitStrategy[ModelHookpointAcausalCrosscoder[Any]]):
    """
    Init strategy that first applies a regular init, and then sets the decoder weight such that each model
    has the same shared decoder weights for the first n_shared_latents.
    """

    def __init__(
        self,
        first_init: InitStrategy[ModelHookpointAcausalCrosscoder[Any]],
        n_shared_latents: int,
    ):
        self.first_init = first_init
        self.n_shared_latents = n_shared_latents

    @torch.no_grad()
    def init_weights(self, cc: ModelHookpointAcausalCrosscoder[Any]) -> None:
        assert cc.W_dec_LMPD.shape[1] == 2, "expected the model dimension to be 2"

        # do the regular init
        self.first_init.init_weights(cc)

        # BUT: sync the shared decoder weights
        cc.W_dec_LMPD[: self.n_shared_latents, 0].copy_(cc.W_dec_LMPD[: self.n_shared_latents, 1])

        assert (cc.W_dec_LMPD[: self.n_shared_latents, 0] == cc.W_dec_LMPD[: self.n_shared_latents, 1]).all()

