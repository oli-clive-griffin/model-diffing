from collections.abc import Iterator

import torch

from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
from crosscode.models.activations.topk import TopkActivation
from crosscode.models.crosslayer_transcoder import CrossLayerTranscoder
from crosscode.models.initialization.init_strategy import InitStrategy
from crosscode.utils import random_direction_init_


class ZeroDecSkipTranscoderInit(InitStrategy[CrossLayerTranscoder[TopkActivation]]):
    def __init__(
        self,
        activation_iterator: Iterator[ModelHookpointActivationsBatch],
        n_samples_for_dec_mean: int,
        enc_init_norm: float,
    ):
        self.activation_iterator = activation_iterator
        self.n_samples_for_dec_mean = n_samples_for_dec_mean
        self.enc_init_norm = enc_init_norm

    @torch.no_grad()
    def init_weights(self, cc: CrossLayerTranscoder[TopkActivation]) -> None:
        assert cc.b_enc_L is not None, "This strategy expects an encoder bias"
        assert cc.b_dec_PD is not None, "This strategy expects a decoder bias"
        assert cc.W_skip_DPD is not None, "This strategy expects a skip connection"

        random_direction_init_(cc.W_enc_DL, self.enc_init_norm)
        cc.b_enc_L.zero_()
        cc.W_dec_LPD.zero_()
        cc.b_dec_PD.copy_(self._get_output_mean_PD())
        cc.W_skip_DPD.zero_()

    def _get_output_mean_PD(self) -> torch.Tensor:
        batches = []
        examples_processed = 0
        while examples_processed < self.n_samples_for_dec_mean:
            acts_BMPD = next(self.activation_iterator).activations_BMPD
            _, n_models, _, _ = acts_BMPD.shape
            assert n_models == 1, "we should have one model"

            # take the output hookpoints
            batches.append(acts_BMPD[:, 0, 1:])
            examples_processed += acts_BMPD.shape[0]

        mean_batches_PD = torch.cat(batches, dim=0).mean(0)
        return mean_batches_PD


class OrthogonalSkipTranscoderInit(InitStrategy[CrossLayerTranscoder[TopkActivation]]):
    def __init__(self, init_norm: float):
        self.init_norm = init_norm
        raise NotImplementedError("not confident in this implementation yet")

    @torch.no_grad()
    def init_weights(self, cc: CrossLayerTranscoder[TopkActivation]) -> None:
        random_direction_init_(cc.W_enc_DL, self.init_norm)

        torch.nn.init.orthogonal_(cc.W_dec_LPD)
        cc.W_dec_LPD.mul_(self.init_norm)

        assert cc.b_enc_L is not None, "This strategy expects an encoder bias"
        cc.b_enc_L.zero_()

        assert cc.b_dec_PD is not None, "This strategy expects a decoder bias"
        cc.b_dec_PD.zero_()

        assert cc.W_skip_DPD is not None, "This strategy expects a skip connection"
        cc.W_skip_DPD.zero_()
