from typing import cast

import torch
from transformer_lens import HookedTransformer  # type: ignore

from model_diffing.scripts.config_common import LLMConfig


def build_llms(
    llms: list[LLMConfig],
    cache_dir: str,
    device: torch.device,
    dtype: str,
) -> list[HookedTransformer]:
    return [
        cast(
            HookedTransformer,  # for some reason, the type checker thinks this is simply an nn.Module
            HookedTransformer.from_pretrained(
                llm.name,
                revision=llm.revision,
                cache_dir=cache_dir,
                dtype=dtype,
            ).to(device),
        )
        for llm in llms
    ]
