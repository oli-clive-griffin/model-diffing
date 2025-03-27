# %%

from collections.abc import Iterator
from itertools import islice
from typing import cast

import torch
from datasets import IterableDataset, load_dataset  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from model_diffing.utils import get_device  # type: ignore

# %%

BASE = "Qwen/Qwen2.5-Math-1.5B"
R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE_DIR = ".cache"
DTYPE = torch.bfloat16

# %%

math = AutoModelForCausalLM.from_pretrained(BASE, cache_dir=CACHE_DIR)
math_tl = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-1.5B",
    hf_model=math,
    dtype=DTYPE,
    cache_dir=CACHE_DIR,
)

r1 = AutoModelForCausalLM.from_pretrained(R1, cache_dir=CACHE_DIR)
r1_tl = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-1.5B",
    hf_model=r1,
    dtype=DTYPE,
    cache_dir=CACHE_DIR,
)

# %%

tok_base = AutoTokenizer.from_pretrained(BASE)
tok_r1 = AutoTokenizer.from_pretrained(R1)

# %%


def iter_questions() -> Iterator[str]:
    DATASET_PATH = "ServiceNow-AI/R1-Distill-SFT"
    DATASET_NAME = "v1"
    SPLIT = "train"  # there's only a train split
    ds = cast(
        IterableDataset,
        load_dataset(
            DATASET_PATH,
            DATASET_NAME,
            split=SPLIT,
            streaming=True,
            cache_dir=CACHE_DIR,
        ),
    )
    for row in ds:
        user_question = row["reannotated_messages"][0]
        assert user_question["role"] == "user"
        yield user_question["content"]


ds = iter_questions()

def format_question(q: str) -> str:
    return f"User: {q}\n\nAssistant: "

def tokenize(tokenizer: PreTrainedTokenizerBase, q: str) -> torch.Tensor:
    return tokenizer.encode(q, return_tensors="pt")  # type: ignore

# %%
device = get_device()

for q in islice(ds, 10):
    print('===============')
    print(q)
    print('== R1')
    toks = tokenize(tokenizer_r1, format_question(q)).to(device)
    output = llm_r1.generate(toks, max_new_tokens=100, return_type="input")[0]
    print(tokenizer_r1.decode(output[toks.shape[0]:]))
    print('== MATH')
    toks = tokenize(tokenizer_base, format_question(q)).to(device)
    output = llm_math.generate(toks, max_new_tokens=100, return_type="input")[0]
    print(tokenizer_base.decode(output[toks.shape[0]:]))
    print('===============')

# %%
