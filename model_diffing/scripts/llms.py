from typing import cast

import torch
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM  # type: ignore

from model_diffing.log import logger
from model_diffing.scripts.config_common import LLMConfig


def build_llms(
    llms: list[LLMConfig],
    cache_dir: str,
    device: torch.device,
    dtype: str,
) -> list[HookedTransformer]:
    llms_out: list[HookedTransformer] = []
    for llm in llms:
        # Create a unique model identifier that will be consistent across Python invocations
        # but unique for different model sources/configurations
        model_identifier_components = []
        
        if llm.name is not None:
            # Case 1: Loading directly from transformer-lens model name
            model_identifier_components.append(f"tl_{llm.name}")
            if llm.revision:
                model_identifier_components.append(llm.revision)
                
            llm_out = cast(
                HookedTransformer,  # for some reason, the type checker thinks this is simply an nn.Module
                HookedTransformer.from_pretrained(
                    llm.name,
                    revision=llm.revision,
                    cache_dir=cache_dir,
                    dtype=dtype,
                ).to(device),
            )
        else:
            # Case 2: Loading from HuggingFace model
            assert llm.base_archicteture_name is not None
            assert llm.hf_model_name is not None
            
            model_identifier_components.append(f"tl_{llm.base_archicteture_name}")
            model_identifier_components.append(f"hf_{llm.hf_model_name}")
            
            logger.info(f"Loading {llm.hf_model_name} from HuggingFace")
            hf_model = AutoModelForCausalLM.from_pretrained(llm.hf_model_name, cache_dir=cache_dir)

            logger.info(
                f"Loading HuggingFace model {llm.hf_model_name} "
                f"into transformer-lens model {llm.base_archicteture_name}"
            )
            llm_out = cast(
                HookedTransformer,
                HookedTransformer.from_pretrained(
                    llm.base_archicteture_name,
                    hf_model=hf_model,
                    cache_dir=cache_dir,
                    dtype=dtype,
                ).to(device),
            )
        
        # Create a deterministic model ID from the component strings
        model_key = "_".join(model_identifier_components)
        
        # Replace any slashes with underscores to avoid potential path issues
        model_key = model_key.replace("/", "_").replace("\\", "_")
        
        # Register the model key as a buffer so it's properly accessible
        # Buffers are persistent state in nn.Module that's not parameters
        llm_out.register_buffer("model_diffing_model_key", 
                               torch.tensor([ord(c) for c in model_key], 
                                           dtype=torch.int64))

        logger.info(f"Assigned model key: {model_key} to model {llm_out.cfg.model_name}")

        llms_out.append(llm_out)

    return llms_out
