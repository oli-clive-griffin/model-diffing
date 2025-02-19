from collections.abc import Iterator
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.topk import TopkActivation
from model_diffing.models.crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import create_cosine_sim_and_relative_norm_histograms
from model_diffing.utils import (
    calculate_explained_variance_X,
    calculate_reconstruction_loss,
    get_explained_var_dict,
    l2_norm,
)


class ZeroDecSkipTranscoderInit(InitStrategy[TopkActivation]):
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


class OrthogonalSkipTranscoderInit(InitStrategy[TopkActivation]):
    def __init__(self, dec_init_norm: float):
        self.dec_init_norm = dec_init_norm

    @torch.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[TopkActivation]) -> None:
        cc.W_enc_XDH[:] = torch.randn_like(cc.W_enc_XDH)
        cc.b_enc_H.zero_()

        torch.nn.init.orthogonal_(cc.W_dec_HXD)
        W_dec_HXD_norm_HX1 = l2_norm(cc.W_dec_HXD, dim=-1, keepdim=True)
        cc.W_dec_HXD.data.div_(W_dec_HXD_norm_HX1)
        cc.W_dec_HXD.data.mul_(self.dec_init_norm)

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

    def _train_step(self, batch_BMPD: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        # hookpoints alternate between input and output
        assert batch_BMPD.shape[2] % 2 == 0, "we should have an even number of hookpoints for this trainer"
        batch_x_BMPiD = batch_BMPD[:, :, ::2]
        batch_y_BMPoD = batch_BMPD[:, :, 1::2]

        train_res = self.crosscoder.forward_train(batch_x_BMPiD)

        reconstruction_loss = calculate_reconstruction_loss(train_res.output_BXD, batch_y_BMPoD)

        reconstruction_loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        self.firing_tracker.add_batch(train_res.hidden_BH.detach().cpu().numpy() > 0)

        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and (self.step + 1) % self.cfg.log_every_n_steps == 0
        ):
            assert batch_y_BMPoD.shape[2] == len(self.hookpoints[1::2])
            assert train_res.output_BXD.shape[2] == len(self.hookpoints[::2])
            names = [" - ".join(pair) for pair in zip(self.hookpoints[::2], self.hookpoints[1::2], strict=True)]

            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_y_BMPoD, train_res.output_BXD),
                ("model", list(range(self.n_models))),
                ("hookpoints", names),
            )

            log_dict: dict[str, Any] = {
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/firing_percentages": self.get_firing_percentage_hist(),
                **explained_variance_dict,
            }

            if self.n_models == 2:
                W_dec_HXD = self.crosscoder.W_dec_HXD.detach().cpu()
                assert W_dec_HXD.shape[1:-1] == (self.n_models, self.n_hookpoints)
                log_dict.update(
                    create_cosine_sim_and_relative_norm_histograms(
                        W_dec_HMPD=W_dec_HXD,
                        hookpoints=self.hookpoints,
                    )
                )

            self.wandb_run.log(log_dict, step=self.step)
