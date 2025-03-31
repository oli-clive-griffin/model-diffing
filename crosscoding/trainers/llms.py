from typing import cast

import torch
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase  # type: ignore

from crosscoding.log import logger
from crosscoding.trainers.config_common import LLMConfig


def build_llms(
    llms: list[LLMConfig],
    cache_dir: str,
    device: torch.device,
    inferenced_type: str,
    # ) -> list[tuple[HookedTransformer, PreTrainedTokenizerBase]]:
) -> list[HookedTransformer]:
    return [build_llm(llm, cache_dir, device, inferenced_type)[0] for llm in llms]


def build_llm(
    llm: LLMConfig,
    cache_dir: str,
    device: torch.device,
    inference_dtype: str,
) -> tuple[HookedTransformer, PreTrainedTokenizerBase]:
    dtype = DTYPE_FROM_STRING[inference_dtype]

    if llm.name is not None:
        model_key = f"tl-{llm.name}"
        if llm.revision:
            model_key += f"_rev-{llm.revision}"

        llm_out = HookedTransformer.from_pretrained_no_processing(
            llm.name,
            revision=llm.revision,
            cache_dir=cache_dir,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llm.name,
            revision=llm.revision,
            cache_dir=cache_dir,
        )
    else:
        assert llm.base_archicteture_name is not None
        assert llm.hf_model_name is not None

        model_key = f"tl-{llm.base_archicteture_name}_hf-{llm.hf_model_name}"

        logger.info(
            f"Loading HuggingFace model {llm.hf_model_name} into transformer-lens model {llm.base_archicteture_name}"
        )

        llm_out = HookedTransformer.from_pretrained_no_processing(
            llm.base_archicteture_name,
            hf_model=AutoModelForCausalLM.from_pretrained(llm.hf_model_name, cache_dir=cache_dir),
            cache_dir=cache_dir,
            dtype=dtype,
        )

    # Replace any slashes with underscores to avoid potential path issues
    model_key = model_key.replace("/", "_").replace("\\", "_")

    # Register the model key as a buffer so it's properly accessible
    # Buffers are persistent state in nn.Module that's not parameters
    llm_out.register_buffer("model_diffing_model_key", torch.tensor([ord(c) for c in model_key], dtype=torch.int64))

    logger.info(f"Assigned model key: {model_key} to model {llm_out.cfg.model_name}")

    return cast(HookedTransformer, llm_out.to(device)), tokenizer


DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
