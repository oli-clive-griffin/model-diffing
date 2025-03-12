from collections.abc import Iterator
from typing import Any

import torch

from model_diffing.models.acausal_crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.models.activations.topk import TopkActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import wandb_histogram
from model_diffing.utils import calculate_reconstruction_loss_summed_MSEs, get_fvu_dict


class ZeroDecSkipTranscoderInit(InitStrategy[AcausalCrosscoder[TopkActivation]]):
    def __init__(self, activation_iterator_BMPD: Iterator[torch.Tensor], n_samples_for_dec_mean: int):
        self.activation_iterator_BMPD = activation_iterator_BMPD
        self.n_samples_for_dec_mean = n_samples_for_dec_mean

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[TopkActivation]) -> None:
        cc.W_enc_XDH[:] = torch.randn_like(cc.W_enc_XDH)
        cc.b_enc_H.zero_()

        cc.W_dec_HXD.zero_()
        cc.b_dec_XD[:] = self._get_output_mean_MPD()

        assert cc.W_skip_XdXd is not None, "W_skip_XdXd should not be None"
        cc.W_skip_XdXd.zero_()

    def _get_output_mean_MPD(self) -> torch.Tensor:
        samples = []
        samples_processed = 0
        while samples_processed < self.n_samples_for_dec_mean:
            sample_BMPD = next(self.activation_iterator_BMPD)
            assert sample_BMPD.shape[2] % 2 == 0, "we should have an even number of hookpoints"
            # get every second hookpoint as output, assuming that these are the output hookpoints
            output_BMPD = sample_BMPD[:, :, 1::2]
            samples.append(output_BMPD)
            samples_processed += output_BMPD.shape[0]

        mean_samples_MPD = torch.cat(samples, dim=0).mean(0)
        return mean_samples_MPD


class OrthogonalSkipTranscoderInit(InitStrategy[AcausalCrosscoder[TopkActivation]]):
    def __init__(self, dec_init_norm: float):
        raise NotImplementedError("not confident in this implementation yet")
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[TopkActivation]) -> None:
        cc.W_enc_XDH.copy_(torch.randn_like(cc.W_enc_XDH))
        cc.b_enc_H.zero_()

        torch.nn.init.orthogonal_(cc.W_dec_HXD)

        assert cc.W_skip_XdXd is not None, "W_skip_XdXd should not be None"
        cc.W_skip_XdXd.zero_()


class TopkSkipTransCrosscoderTrainer(BaseModelHookpointTrainer[BaseTrainConfig, TopkActivation]):
    # shapes:
    # B: batch size
    # M: number of models
    # P: number of hookpoints
    # Pi: number of input hookpoints
    # Po: number of output hookpoints (should be equal to Pi)
    # D: activation dimension (in this case, d_mlp)

    def _calculate_loss_and_log(
        self,
        batch_BMPD: torch.Tensor,
        train_res: AcausalCrosscoder.ForwardResult,
        log: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        assert batch_BMPD.shape[2] % 2 == 0, "we should have an even number of hookpoints for this trainer"
        batch_x_BMPiD = batch_BMPD[:, :, ::2]
        batch_y_BMPoD = batch_BMPD[:, :, 1::2]

        train_res = self.crosscoder.forward_train(batch_x_BMPiD)
        self.firing_tracker.add_batch(train_res.hidden_BH)

        loss = calculate_reconstruction_loss_summed_MSEs(train_res.recon_acts_BXD, batch_y_BMPoD)

        if log:
            assert batch_y_BMPoD.shape[2] == len(self.hookpoints[1::2])
            assert train_res.recon_acts_BXD.shape[2] == len(self.hookpoints[::2])
            log_dict: dict[str, Any] = {
                "train/loss": loss.item(),
                **self._get_fvu_dict(batch_y_BMPoD, train_res.recon_acts_BXD),
            }
            return loss, log_dict

        return loss, None

    def _get_fvu_dict(self, batch_BMPD: torch.Tensor, recon_acts_BMPD: torch.Tensor) -> dict[str, float]:
        names = [" - ".join(pair) for pair in zip(self.hookpoints[::2], self.hookpoints[1::2], strict=True)]
        return get_fvu_dict(
            batch_BMPD,
            recon_acts_BMPD,
            ("model", list(range(self.n_models))),
            ("hookpoints", names),
        )
