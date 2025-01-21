from typing import cast

import torch
from transformer_lens import HookedTransformer

from model_diffing.scripts.config_common import LLMsConfig


def build_llms(llms: LLMsConfig, cache_dir: str, device: torch.device) -> list[HookedTransformer]:
    return [
        cast(
            HookedTransformer,  # for some reason, the type checker thinks this is simply an nn.Module
            HookedTransformer.from_pretrained(
                llm.name,
                revision=llm.revision,
                cache_dir=cache_dir,
                dtype=llms.inference_dtype,
            ).to(device),
        )
        for llm in llms.models
    ]
