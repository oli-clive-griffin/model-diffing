# from pathlib import Path
# from typing import Any

# import torch

# from crosscode.data.activations_dataloader import ModelHookpointActivationsBatch
# from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder
# from crosscode.models.activations.jumprelu import AnthropicSTEJumpReLUActivation
# from crosscode.models.activations.topk import BatchTopkActivation, GroupMaxActivation, TopkActivation
# from crosscode.trainers.base_diffing_trainer import ModelDiffingWrapper
# from crosscode.trainers.firing_tracker import FiringTracker
# from crosscode.trainers.jan_update_crosscoder.trainer import pre_act_loss, tanh_sparsity_loss
# from crosscode.trainers.topk_crosscoder.config import TopKTrainConfig
# from crosscode.trainers.topk_crosscoder.trainer import aux_loss
# from crosscode.trainers.utils import get_l0_stats, wandb_histogram
# from crosscode.utils import (
#     calculate_reconstruction_loss_summed_norm_MSEs,
#     get_fvu_dict,
#     get_summed_decoder_norms_L,
#     not_none,
# )


# class TopKModelDiffingFebUpdateWrapper(ModelDiffingWrapper[TopkActivation | BatchTopkActivation | GroupMaxActivation]):
#     def __init__(
#         self,
#         model: ModelHookpointAcausalCrosscoder[TopkActivation | BatchTopkActivation | GroupMaxActivation],
#         n_shared_latents: int,
#         hookpoints: list[str],
#     ):
#         self.firing_tracker = FiringTracker(activation_size=model.n_latents, device=self.device)
#         self.model = model
#         self.n_shared_latents = n_shared_latents
#         self.hookpoints = hookpoints

#     def run_batch(
#         self,
#         step: int,
#         batch: ModelHookpointActivationsBatch,
#         log: bool,
#     ) -> tuple[torch.Tensor, dict[str, float] | None]:
#         train_res = self.model.forward_train(batch.activations_BMPD)
#         self.firing_tracker.add_batch(train_res.latents_BL)

#         reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(
#             batch.activations_BMPD, train_res.recon_acts_BMPD
#         )

#         loss = reconstruction_loss

#         if log:
#             hidden_shared_BLs = train_res.latents_BL[:, : self.n_shared_latents]
#             hidden_indep_BLi = train_res.latents_BL[:, self.n_shared_latents :]

#             log_dict: dict[str, float] = {
#                 "train/reconstruction_loss": reconstruction_loss.item(),
#                 "train/tanh_sparsity_loss_shared": tanh_sparsity_loss_shared.item(),
#                 "train/tanh_sparsity_loss_indep": tanh_sparsity_loss_indep.item(),
#                 "train/pre_act_loss": pre_act_loss.item(),
#                 "train/loss": loss.item(),
#                 **self._get_fvu_dict(batch.activations_BMPD, train_res.recon_acts_BMPD),
#                 **get_l0_stats(hidden_shared_BLs, name="shared_l0"),
#                 **get_l0_stats(hidden_indep_BLi, name="indep_l0"),
#                 **get_l0_stats(train_res.latents_BL, name="both_l0"),
#             }
#             return loss, log_dict

#         return loss, None

#     def save(self, step: int) -> Path:
#         checkpoint_path = self.save_dir / f"step_{step}"
#         self.model.with_folded_scaling_factors(self.scaling_factors_MP).save(checkpoint_path)
#         return checkpoint_path

#     def _get_fvu_dict(self, y_BPD: torch.Tensor, recon_y_BPD: torch.Tensor) -> dict[str, float]:
#         return get_fvu_dict(
#             y_BPD,
#             recon_y_BPD,
#             ("model", ["0", "1"]),
#         )

#     def logs(self, step: int, include_expensive_logs: bool = False) -> dict[str, Any]:
#         ...


# class TopKFebUpdateDiffingTrainer(
#     BaseFebUpdateDiffingTrainer[TopKTrainConfig, TopkActivation | GroupMaxActivation | BatchTopkActivation]
# ):
#     def _calculate_loss_and_log(
#         self,
#         batch_BMPD: torch.Tensor,
#         train_res: ModelHookpointAcausalCrosscoder.ForwardResult,
#         log: bool,
#     ) -> tuple[torch.Tensor, dict[str, Any] | None]:
#         reconstruction_loss = calculate_reconstruction_loss_summed_norm_MSEs(batch_BMPD, train_res.recon_acts_BMPD)
#         aux_loss = self.aux_loss(batch_BMPD, train_res)
#         loss = reconstruction_loss + self.cfg.lambda_aux * aux_loss

#         if log:
#             log_dict: dict[str, Any] = {
#                 "train/loss": loss.item(),
#                 "train/reconstruction_loss": reconstruction_loss.item(),
#                 "train/aux_loss": aux_loss.item(),
#                 **self._get_fvu_dict(batch_BMPD, train_res.recon_acts_BMPD),
#             }

#             return reconstruction_loss, log_dict

#         return reconstruction_loss, None

#     def aux_loss(
#         self, batch_BMPD: torch.Tensor, train_res: ModelHookpointAcausalCrosscoder.ForwardResult
#     ) -> torch.Tensor:
#         """train to reconstruct the error with the topk dead latents"""
#         return aux_loss(
#             pre_activations_BL=train_res.pre_activations_BL,
#             dead_features_mask_L=self.firing_tracker.tokens_since_fired_L > self.cfg.dead_latents_threshold_n_examples,
#             k_aux=not_none(self.cfg.k_aux),
#             decode_BXD=self.model.decode_BMPD,
#             error_BXD=batch_BMPD - train_res.recon_acts_BMPD,
#         )
