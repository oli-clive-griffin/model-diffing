# import fire  # type: ignore

# from model_diffing.data.model_hookpoint_dataloader import build_dataloader
# from model_diffing.log import logger
# from model_diffing.models.crosscoder import ModelHookpointAcausalCrosscoder
# from model_diffing.models.activations.topk import TopkActivation
# from model_diffing.scripts.base_trainer import run_exp
# from model_diffing.scripts.llms import build_llms
# from model_diffing.scripts.skip_trans_crosscoder.config import TopkSkipTransCrosscoderExperimentConfig
# from model_diffing.scripts.skip_trans_crosscoder.trainer import (
#     TopkSkipTransCrosscoderTrainer,
#     ZeroDecSkipTranscoderInit,
# )
# from model_diffing.scripts.utils import build_wandb_run
# from model_diffing.utils import get_device


# def build_trainer(cfg: TopkSkipTransCrosscoderExperimentConfig) -> TopkSkipTransCrosscoderTrainer:
#     device = get_device()

#     llms = build_llms(
#         cfg.data.activations_harvester.llms,
#         cfg.cache_dir,
#         device,
#         inferenced_type=cfg.data.activations_harvester.inference_dtype,
#     )

#     hookpoints = [
#         f"blocks.{block_idx}.mlp.{pos}"  #
#         for block_idx in cfg.mlp_indices
#         for pos in ["hook_pre", "hook_post"]
#     ]

#     dataloader = build_dataloader(
#         cfg=cfg.data,
#         llms=llms,
#         hookpoints=hookpoints,
#         batch_size=cfg.train.minibatch_size(),
#         cache_dir=cfg.cache_dir,
#     )

#     # HACK: need to average over each consecutive pair of hookpoints
#     assert len(hookpoints) == 2, "this hack only supports 2 hookpoints"
#     dataloader._norm_scaling_factors_MP = dataloader._norm_scaling_factors_MP.mean(dim=1, keepdim=True)

#     # for each pair of hookpoints, only the input hook is passed, though the model, the other acts as the label
#     # it's always even because we alternate between input and output above
#     hookpoints_in_out = len(hookpoints) // 2
#     crosscoding_dims = (len(llms), hookpoints_in_out)

#     crosscoder = ModelHookpointAcausalCrosscoder(
#         crosscoding_dims=crosscoding_dims,
#         d_model=llms[0].cfg.d_mlp,
#         n_latents=cfg.crosscoder.n_latents,
#         init_strategy=ZeroDecSkipTranscoderInit(
#             activation_iterator_BMPD=dataloader.get_activations_iterator_BMPD(),
#             n_samples_for_dec_mean=100_000,
#             enc_init_norm=0.1,  # cfg.crosscoder.enc_init_norm,
#         ),
#         activation_fn=TopkActivation(k=cfg.crosscoder.k),
#         skip_linear=True,
#         use_encoder_bias=cfg.crosscoder.use_encoder_bias,
#         use_decoder_bias=cfg.crosscoder.use_decoder_bias,
#     )

#     crosscoder = crosscoder.to(device)

#     wandb_run = build_wandb_run(cfg)

#     return TopkSkipTransCrosscoderTrainer(
#         cfg=cfg.train,
#         activations_dataloader=dataloader,
#         crosscoder=crosscoder,
#         wandb_run=wandb_run,
#         device=device,
#         hookpoints=hookpoints,
#         save_dir=cfg.save_dir,
#     )


# if __name__ == "__main__":
#     logger.info("Starting...")
#     fire.Fire(run_exp(build_trainer, TopkSkipTransCrosscoderExperimentConfig))
