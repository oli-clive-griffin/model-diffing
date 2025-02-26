# %%
import asyncio
import concurrent.futures
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import Thread
from typing import Any, cast

import anthropic
import torch
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.models.qwen2 import Qwen2ForCausalLM

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.model_hookpoint_dataloader import ScaledModelHookpointActivationsDataloader
from model_diffing.data.token_loader import MathDatasetTokenSequenceLoader
from model_diffing.interp import (
    gather_max_activating_examples,
    iterate_activations_with_text,
)
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateTrainConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import JanUpdateCrosscoderTrainer

# %%

BASE = "Qwen/Qwen2.5-Math-1.5B"
R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

llm_math_hf = AutoModelForCausalLM.from_pretrained(BASE, cache_dir=".cache")
llm_r1_hf = AutoModelForCausalLM.from_pretrained(R1, cache_dir=".cache")

# %%

# this should work because the model is the same, just a finetune
llm_math = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", hf_model=llm_math_hf)
llm_r1 = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", hf_model=llm_r1_hf)


# %%

llm_r1.tokenizer.padding_side
# %%
llm_r1.tokenizer.special_tokens_map

# %%
llm_math.tokenizer.pad_token
# %%


# %%

tokenizer_base = AutoTokenizer.from_pretrained(BASE)
tokenizer_r1 = AutoTokenizer.from_pretrained(R1)

# %%

token_sequence_loader = MathDatasetTokenSequenceLoader(
    tokenizer=tokenizer_r1,
    batch_size=16,
    max_sequence_length=1024,
    cache_dir=".cache",
)

for asdf in token_sequence_loader.get_sequences_batch_iterator():
    print(asdf.tokens_BS.shape)

# %%

activations_harvester = ActivationsHarvester(
    llms=[llm_math, llm_r1],
    hookpoints=["blocks.0.hook_resid_post"],
)

dl = ScaledModelHookpointActivationsDataloader(
    token_sequence_loader=token_sequence_loader,
    activations_harvester=activations_harvester,
    activations_shuffle_buffer_size=1000,
    yield_batch_size=100,
    n_tokens_for_norm_estimate=10,
)

# for asdf in dl.get_shuffled_activations_iterator_BMPD():
#     print(asdf.shape)

# %%


llm_r1.forward(
    torch.tensor([[1, 2, 3]]),
    return_type="logits",
    attention_mask=torch.tensor([[False, True, True]]),
)

# %%
