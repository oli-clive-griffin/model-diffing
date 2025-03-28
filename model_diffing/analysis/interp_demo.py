# %%
import os

# %%
from pathlib import Path
from typing import cast

import torch
import yaml  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,  # type: ignore
)

from model_diffing.analysis import metrics, visualization
from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.token_loader import MathDatasetTokenSequenceLoader
from model_diffing.interp import (
    topk_seqs_table,
    gather_latent_summaries,
)
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateExperimentConfig
from model_diffing.scripts.wandb_scripts.main import download_experiment_checkpoint
from model_diffing.utils import get_device

# %%

BASE = "Qwen/Qwen2.5-Math-1.5B"
R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE_DIR = ".cache"
DTYPE = torch.bfloat16

# %%

llm_math = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-1.5B",
    hf_model=AutoModelForCausalLM.from_pretrained(BASE, cache_dir=CACHE_DIR),
    dtype=DTYPE,
    cache_dir=CACHE_DIR,
)

llm_r1 = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-1.5B",
    hf_model=AutoModelForCausalLM.from_pretrained(R1, cache_dir=CACHE_DIR),
    dtype=DTYPE,
    cache_dir=CACHE_DIR,
)
# %% Setup

print(f"Current working directory: {Path.cwd()}")
os.chdir(Path(__file__).parent.parent.parent)
print(f"Current working directory: {Path.cwd()}")

device = get_device()
cache_dir = ".cache"

DOWNLOAD_DIR = ".data/local"
download_experiment_checkpoint(
    run_id="gmiyehyq",
    version="v6",
    destination_dir=DOWNLOAD_DIR,
)

# %%

sae = DiffingCrosscoder.load(Path(DOWNLOAD_DIR) / "model").to(device)
assert sae.is_folded.item()
sae.make_decoder_max_unit_norm_()

with open(Path(DOWNLOAD_DIR) / "experiment_config.yaml") as f:
    exp_config = JanUpdateExperimentConfig(**yaml.safe_load(f))

# %%

assert sae.W_dec_HXD.shape[1:-1] == (2, 1)  # two models, single hookpoint
W_dec_HMD = sae.W_dec_HXD[:, :, 0]  # remove the hookpoint dimension

m1_W_dec_HD, m2_W_dec_HD = W_dec_HMD.unbind(dim=1)
relative_decoder_norms_H = metrics.compute_relative_norms_N(m1_W_dec_HD, m2_W_dec_HD)
visualization.relative_norms_hist(relative_decoder_norms_H, title="Relative norms of decoder vectors").show()

shared_latent_mask = metrics.get_shared_latent_mask(relative_decoder_norms_H, min_thresh=0.25, max_thresh=0.5)
cosine_sims = metrics.compute_cosine_similarities_N(m1_W_dec_HD, m2_W_dec_HD)[shared_latent_mask]
visualization.plot_cosine_sim(
    cosine_sims, title="Cosine similarity of decoder vectors (where 0.25 < relative_norm < 0.5)"
).show()

# %%

llms = build_llms(
    exp_config.data.activations_harvester.llms,
    cache_dir=cache_dir,
    inferenced_type=exp_config.data.activations_harvester.inference_dtype,
    device=device,
)

# %%
tokenizer = cast(PreTrainedTokenizerBase, llms[0].tokenizer)
if not isinstance(tokenizer, PreTrainedTokenizerBase):
    raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

assert exp_config.data.token_sequence_loader.type == "MathDatasetTokenSequenceLoader"

# %%


examples_iterator = iterate_activations_with_text(
    token_sequence_loader=MathDatasetTokenSequenceLoader(
        tokenizer=tokenizer,
        max_sequence_length=exp_config.data.token_sequence_loader.max_sequence_length,
        cache_dir=cache_dir,
        batch_size=2,
    ),
    activations_harvester=ActivationsHarvester(
        llms=llms,
        hookpoints=exp_config.hookpoints,
    ),
)

# %% Interp

examples_by_latent = gather_latent_summaries(
    examples_iterator,
    cc=sae,
    total_batches=4,
    context_size=20,
    topk_per_latent_per_batch=5,
    latents_to_inspect=list(range(100)),
)

# %%

for summary in examples_by_latent.values():
    if len(summary.selected_examples) == 0:
        continue

    relative_decoder_norm = relative_decoder_norms_H[summary.index].item()
    if 0.4 < relative_decoder_norm < 0.6:
        continue

    topk_seqs_table(
        summary,
        tokenizer,
        relative_decoder_norm,
        topk=30,
    )


# %%

# ==============================
# model-aligned latents
# ==============================

model_1_topk = relative_decoder_norms_H.topk(100, largest=True)
model_2_topk = relative_decoder_norms_H.topk(100, largest=False)

model_aligned_latent_indices = torch.cat([model_1_topk.indices, model_2_topk.indices])
model_aligned_latent_vals = torch.cat([model_1_topk.values, model_2_topk.values])
# %%


torch.set_printoptions(precision=3, sci_mode=False)
print(model_aligned_latent_indices)
print(model_aligned_latent_vals)
# reset print options
torch.set_printoptions()

# %%

info = torch.cuda.mem_get_info()
print(f"GPU memory: {info[0] / 1024**2} MB used, {info[1] / 1024**2} MB total")

print(f"gathering {len(model_aligned_latent_indices)} latents")
model_aligned_examples_by_latent = gather_latent_summaries(
    examples_iterator,
    cc=sae,
    total_batches=3000,
    context_size=20,
    topk_per_latent_per_batch=5,
    latents_to_inspect=model_aligned_latent_indices.tolist(),
)

# %%

for summary in model_aligned_examples_by_latent.values():
    if len(summary.selected_examples) == 0:
        continue

    relative_decoder_norm = relative_decoder_norms_H[summary.index].item()
    if 0.4 < relative_decoder_norm < 0.6:
        continue

    topk_seqs_table(summary, tokenizer, relative_decoder_norm)

# %%
