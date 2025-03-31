# from crosscoding.data.activations_dataloader import (
#     ActivationsHarvester,
#     ModelHookpointActivationsDataloader,
#     TokenSequenceLoader,
# )
# from crosscoding.llms import build_llms
# from crosscoding.models import ModelHookpointAcausalCrosscoder, ReLUActivation
# from crosscoding.trainers.config_common import (
#     ActivationsHarvesterConfig,
#     DataConfig,
#     HuggingfaceTextDatasetConfig,
#     LLMConfig,
# )
# from crosscoding.trainers.train_l1_crosscoder.trainer import L1CrosscoderTrainer

# sae = ModelHookpointAcausalCrosscoder(
#     d_model=1024,
#     n_latents=16_384,
#     activation_fn=ReLUActivation(),
#     use_encoder_bias=True,
#     use_decoder_bias=True,
# )

# # data_cfg = DataConfig(
# #     token_sequence_loader=HuggingfaceTextDatasetConfig(
# #         hf_dataset_name="monology/pile-uncopyrighted",
# #         sequence_length=1024,
# #         shuffle_buffer_size=100_000,
# #     ),
# #     activations_harvester=ActivationsHarvesterConfig(
# #         llms=[
# #             LLMConfig(
# #                 name="meta-llama/Meta-Llama-3-8B-Instruct",
# #                 revision="main",
# #             )
# #         ],
# #         cache_mode="no_cache",
# #         harvesting_batch_size=1024,
# #     ),
# # )
# cache_dir = ".cache"

# device = get_device()

# llms = build_llms(
#     data_cfg.activations_harvester.llms,
#     cache_dir,
#     device,
#     inferenced_type=data_cfg.activations_harvester.inference_dtype,
# )

# # first, get an iterator over sequences of tokens
# token_sequence_loader = TokenSequenceLoader(
#     hf_dataset_name="",
#     sequence_length=2048,
#     cache_dir=cache_dir,
#     tokenizer=llms[0].tokenizer,
#     batch_size=cfg.activations_harvester.harvesting_batch_size,
# )

# # Create activations cache directory if cache is enabled
# activations_cache_dir = None
# if cfg.activations_harvester.cache_mode != "no_cache":
#     activations_cache_dir = os.path.join(cache_dir, "activations_cache")
#     logger.info(f"Activations will be cached in: {activations_cache_dir}")

# # then, run these sequences through the model to get activations
# activations_harvester = ActivationsHarvester(
#     llms=llms,
#     hookpoints=hookpoints,
#     cache_dir=activations_cache_dir,
#     cache_mode=cfg.activations_harvester.cache_mode,
# )

# activations_dataloader = ModelHookpointActivationsDataloader(
#     token_sequence_loader=token_sequence_loader,
#     activations_harvester=activations_harvester,
#     yield_batch_size_B=1024,
#     n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
# )


# trainer = L1CrosscoderTrainer(
#     cfg=None,
#     activations_dataloader=activations_dataloader,
#     crosscoder=sae,
#     wandb_run=None,
#     device=None,
#     save_dir=None,
#     crosscoding_dims=None,
# )
