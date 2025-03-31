# from collections.abc import Iterator
# from typing import Any

# import torch

# from model_diffing.models.activations.topk import TopkActivation
# from model_diffing.models.crosscoder import ModelHookpointAcausalCrosscoder, InitStrategy
# from model_diffing.scripts.base_trainer import BaseModelHookpointAcausalTrainer
# from model_diffing.scripts.config_common import BaseTrainConfig
# from model_diffing.utils import calculate_reconstruction_loss_summed_norm_MSEs, get_fvu_dict, random_direction_init_


# class ZeroDecSkipTranscoderInit(InitStrategy[ModelHookpointAcausalCrosscoder[TopkActivation]]):
#     def __init__(
#         self,
#         activation_iterator_BMPD: Iterator[torch.Tensor],
#         n_samples_for_dec_mean: int,
#         enc_init_norm: float,
#     ):
#         self.activation_iterator_BMPD = activation_iterator_BMPD
#         self.n_samples_for_dec_mean = n_samples_for_dec_mean
#         self.enc_init_norm = enc_init_norm

#     @torch.no_grad()
#     def init_weights(self, cc: ModelHookpointAcausalCrosscoder[TopkActivation]) -> None:
#         random_direction_init_(cc.W_enc_XDL, self.enc_init_norm)
#         cc.W_dec_LXoDo.zero_()

#         assert cc.b_enc_L is not None, "This strategy expects an encoder bias"
#         cc.b_enc_L.zero_()

#         assert cc.b_dec_XD is not None, "This strategy expects a decoder bias"
#         cc.b_dec_XD[:] = self._get_output_mean_MPD()

#         assert cc.W_skip_XDD is not None, "This strategy expects a skip connection"
#         cc.W_skip_XDD.zero_()

#     def _get_output_mean_MPD(self) -> torch.Tensor:
#         samples = []
#         samples_processed = 0
#         while samples_processed < self.n_samples_for_dec_mean:
#             sample_BMPD = next(self.activation_iterator_BMPD)
#             assert sample_BMPD.shape[2] % 2 == 0, "we should have an even number of hookpoints"
#             # get every second hookpoint as output, assuming that these are the output hookpoints
#             output_BMPD = sample_BMPD[:, :, 1::2]
#             samples.append(output_BMPD)
#             samples_processed += output_BMPD.shape[0]

#         mean_samples_MPD = torch.cat(samples, dim=0).mean(0)
#         return mean_samples_MPD


# class OrthogonalSkipTranscoderInit(InitStrategy[ModelHookpointAcausalCrosscoder[TopkActivation]]):
#     def __init__(self, init_norm: float):
#         self.init_norm = init_norm
#         raise NotImplementedError("not confident in this implementation yet")

#     @torch.no_grad()
#     def init_weights(self, cc: ModelHookpointAcausalCrosscoder[TopkActivation]) -> None:
#         random_direction_init_(cc.W_enc_XDL, self.init_norm)

#         torch.nn.init.orthogonal_(cc.W_dec_LXoDo)
#         cc.W_dec_LXoDo.mul_(self.init_norm)

#         assert cc.b_enc_L is not None, "This strategy expects an encoder bias"
#         cc.b_enc_L.zero_()

#         assert cc.b_dec_XD is not None, "This strategy expects a decoder bias"
#         cc.b_dec_XD.zero_()

#         assert cc.W_skip_XDD is not None, "This strategy expects a skip connection"
#         cc.W_skip_XDD.zero_()


# class TopkSkipTransCrosscoderTrainer(BaseModelHookpointAcausalTrainer[BaseTrainConfig, TopkActivation]):
#     # shapes:
#     # B: batch size
#     # M: number of models
#     # P: number of hookpoints
#     # Pi: number of input hookpoints
#     # Po: number of output hookpoints (should be equal to Pi)
#     # D: activation dimension (in this case, d_mlp)
#     def __init__(self, *args: Any, **kwargs: Any):
#         raise NotImplementedError("This trainer does not have auxiliary losses implemented yet")

#     def _calculate_loss_and_log(
#         self,
#         batch_BMPD: torch.Tensor,
#         train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
#         log: bool,
#     ) -> tuple[torch.Tensor, dict[str, float] | None]:
#         assert batch_BMPD.shape[2] % 2 == 0, "we should have an even number of hookpoints for this trainer"
#         batch_x_BMPiD = batch_BMPD[:, :, ::2]
#         batch_y_BMPoD = batch_BMPD[:, :, 1::2]

#         train_res = self.crosscoder.forward_train(batch_x_BMPiD)
#         self.firing_tracker.add_batch(train_res.latents_BL)

#         loss = calculate_reconstruction_loss_summed_norm_MSEs(train_res.recon_acts_BXD, batch_y_BMPoD)

#         if log:
#             assert batch_y_BMPoD.shape[2] == len(self.hookpoints[1::2])
#             assert train_res.recon_acts_BXD.shape[2] == len(self.hookpoints[::2])
#             log_dict: dict[str, Any] = {
#                 "train/loss": loss.item(),
#                 **self._get_fvu_dict(batch_y_BMPoD, train_res.recon_acts_BXD),
#             }
#             return loss, log_dict

#         return loss, None

#     def _get_fvu_dict(self, batch_BMPD: torch.Tensor, recon_acts_BMPD: torch.Tensor) -> dict[str, float]:
#         names = [" - ".join(pair) for pair in zip(self.hookpoints[::2], self.hookpoints[1::2], strict=True)]
#         return get_fvu_dict(
#             batch_BMPD,
#             recon_acts_BMPD,
#             ("model", list(range(self.n_models))),
#             ("hookpoints", names),
#         )
