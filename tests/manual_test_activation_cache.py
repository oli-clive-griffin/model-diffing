# """Test script for the ActivationsCache functionality. Needs to be run manually."""

# import shutil
# import tempfile
# import time
# from itertools import islice
# from pathlib import Path
# from typing import cast

# import torch

# from crosscoding.data.activations_dataloader import ScaledModelHookpointActivationsDataloader
# from crosscoding.data.token_loader import ToyOverfittingTokenSequenceLoader
# from crosscoding.llms import build_llms
# from crosscoding.trainers.config_common import LLMConfig


# def main():
#     # Create a temporary directory for cache
#     temp_dir = Path(tempfile.mkdtemp())
#     try:
#         print(f"Using temporary cache directory: {temp_dir}")

#         # Create toy model configs
#         llm_config1 = LLMConfig(name="EleutherAI/pythia-160m", revision="step142000")
#         llm_config2 = LLMConfig(name="EleutherAI/pythia-160m", revision="step143000")

#         # Build models
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         models = build_llms(
#             [llm_config1, llm_config2],
#             cache_dir=".cache",
#             device=device,
#             inferenced_type="float32",
#         )
#         model1, model2 = models

#         # Verify they have different model keys
#         model1_key = "".join(chr(i) for i in cast(torch.Tensor, model1.model_diffing_model_key).tolist())
#         model2_key = "".join(chr(i) for i in cast(torch.Tensor, model2.model_diffing_model_key).tolist())  # type: ignore
#         print(f"Model 1 key: {model1_key}")
#         print(f"Model 2 key: {model2_key}")

#         # Create activation cache
#         cache_dir = temp_dir / "activation_cache"

#         # Create hook point names
#         hook_points = ["blocks.8.hook_resid_post"]

#         # Test with first model
#         print("\n=== Testing with Model 1 ===")

#         def build_dataloader():
#             return ScaledModelHookpointActivationsDataloader(
#                 token_sequence_loader=ToyOverfittingTokenSequenceLoader(
#                     sequence_length=16,
#                     vocab_size=100,
#                     first_n_tokens_special=2,
#                     batch_size=4,
#                 ),
#                 activations_harvester=ActivationsHarvester(
#                     llms=[model1],
#                     hookpoints=hook_points,
#                     cache_dir=str(cache_dir),
#                     cache_mode="cache",
#                 ),
#                 yield_batch_size_B=4,
#                 n_tokens_for_norm_estimate=16,
#             )

#         # First iteration (no cache)
#         print("First iteration (fill cache):")
#         start_time = time.time()
#         for _batch in islice(build_dataloader().get_activations_iterator_BMPD(), 1000):
#             pass
#         first_iter_time = time.time() - start_time
#         print(f"First iteration took {first_iter_time:.4f} seconds")

#         # Second iteration (with cache)
#         print("\nSecond iteration (with cache):")
#         start_time = time.time()
#         for _batch in islice(build_dataloader().get_activations_iterator_BMPD(), 1000):
#             pass
#         second_iter_time = time.time() - start_time
#         print(f"Second iteration took {second_iter_time:.4f} seconds")

#         print(f"Speedup factor: {first_iter_time / second_iter_time:.2f}x")

#     finally:
#         # Clean up the temporary directory
#         shutil.rmtree(temp_dir)
#         print(f"\nCleaned up temporary directory: {temp_dir}")


# if __name__ == "__main__":
#     main()
