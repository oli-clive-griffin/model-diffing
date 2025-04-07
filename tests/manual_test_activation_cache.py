"""Test script for the ActivationsCache functionality. Needs to be run manually."""

import shutil
import time
from itertools import islice
from pathlib import Path
from typing import cast

import torch

from crosscode.data.activations_dataloader import build_model_hookpoint_dataloader
from crosscode.llms import build_llms
from crosscode.saveable_module import DTYPE_TO_STRING
from crosscode.trainers.config_common import (
    ActivationsHarvesterConfig,
    DataConfig,
    HuggingfaceTextDatasetConfig,
    LLMConfig,
)
from crosscode.utils import get_device

CACHE_DIR = ".cache"


def test():
    print(f"Using cache directory: {CACHE_DIR}")

    # Create toy model configs
    llm_config1 = LLMConfig(name="EleutherAI/pythia-160m", revision="step142000")
    llm_config2 = LLMConfig(name="EleutherAI/pythia-160m", revision="step143000")

    # Create hook point names
    hook_points = ["blocks.8.hook_resid_post"]

    # Build models
    device = get_device()
    models = build_llms(
        [llm_config1, llm_config2],
        cache_dir=CACHE_DIR,
        device=device,
        inferenced_type=DTYPE_TO_STRING[torch.float32],
    )

    model1, model2 = models

    def build_dataloader():
        data_cfg = DataConfig(
            activations_harvester=ActivationsHarvesterConfig(
                llms=[llm_config1, llm_config2],
                cache_mode="cache",
                harvesting_batch_size=1,
            ),
            n_tokens_for_norm_estimate=1,
            token_sequence_loader=HuggingfaceTextDatasetConfig(
                sequence_length=128,
            ),
        )

        return build_model_hookpoint_dataloader(
            batch_size=1,
            cache_dir=CACHE_DIR,
            hookpoints=hook_points,
            llms=models,
            cfg=data_cfg,
        )

        # Verify they have different model keys

    model1_key = "".join(chr(i) for i in cast(torch.Tensor, model1.crosscode_model_key).tolist())
    model2_key = "".join(chr(i) for i in cast(torch.Tensor, model2.crosscode_model_key).tolist())  # type: ignore
    print(f"Model 1 key: {model1_key}")
    print(f"Model 2 key: {model2_key}")
    dl1 = build_dataloader()
    dl2 = build_dataloader()

    print("First iteration (fill cache):")
    start_time = time.time()
    for _ in islice(dl1.get_activations_iterator(), 1000):
        pass
    first_iter_time = time.time() - start_time
    print(f"First iteration took {first_iter_time:.4f} seconds")

    # Second iteration (with cache)
    print("\nSecond iteration (with cache):")
    start_time = time.time()
    for _ in islice(dl2.get_activations_iterator(), 1000):
        pass
    second_iter_time = time.time() - start_time
    print(f"Second iteration took {second_iter_time:.4f} seconds")

    print(f"Speedup factor: {first_iter_time / second_iter_time:.2f}x")


def main():
    try:
        return test()
    finally:
        activations_cache_dir = Path(CACHE_DIR) / "activations_cache"
        shutil.rmtree(activations_cache_dir)
        print(f"\nCleaned up activations cache directory: {activations_cache_dir}")


if __name__ == "__main__":
    main()
