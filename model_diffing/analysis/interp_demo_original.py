# %%
import sys
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm import tqdm  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformer_lens.hook_points import HookPoint  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.token_loader import (
    MathDatasetConfig,
    build_tokens_sequence_loader,
)
from model_diffing.data.token_loader.base import TokenSequenceLoader
from model_diffing.interp import (
    ActivationsWithText,
    LatentExaminer,
    get_relative_decoder_norms_H,
    topk_seqs_table,
)
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateExperimentConfig
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.wandb_scripts.main import download_experiment_checkpoint
from model_diffing.utils import get_device

# %% Setup

print(f"Current working directory: {Path.cwd()}")
os.chdir(Path(__file__).parent.parent.parent)
print(f"Current working directory: {Path.cwd()}")

device = get_device()
cache_dir = ".cache"

path = download_experiment_checkpoint(
    # this run is a layer-8 jumprelu SAE on pythia-160m for ≈ 90m tokens
    # tokens = (checkpoint_idx * checkpoint_every_n_batches * batch_size)
    #        = (12 * 1000 * 8192)
    run_id="24bnrl1g",
    version="v1",
    destination_dir=".data/artifact_download",
)

# %%
cc: AcausalCrosscoder[Any] = AcausalCrosscoder.load(path / "model")
assert cc.is_folded.item()
cc.to(device)

with open(path / "experiment_config.yaml") as f:
    exp_config = JumpReLUModelDiffingFebUpdateExperimentConfig(**yaml.safe_load(f))

llms = build_llms(
    exp_config.data.activations_harvester.llms,
    cache_dir=cache_dir,
    device=device,
    inferenced_type="bfloat16",
)

assert isinstance(exp_config.data.token_sequence_loader, MathDatasetConfig)


tokenizer: Any = llms[0].tokenizer  # type: ignore

# %%

relative_decoder_norms_H = get_relative_decoder_norms_H(cc.with_decoder_unit_norm())
plt.hist(relative_decoder_norms_H.detach().cpu().numpy(), bins=100, density=True, log=True)
plt.show()
# %%

token_sequence_loader = build_tokens_sequence_loader(
    cfg=exp_config.data.token_sequence_loader,
    cache_dir=cache_dir,
    tokenizer=llms[0].tokenizer,  # type: ignore
    batch_size=exp_config.data.activations_harvester.harvesting_batch_size,
)

activations_harvester = ActivationsHarvester(
    llms=llms,
    hookpoints=[exp_config.hookpoint],
)
# %%


def iterate_activations_with_text(
    token_sequence_loader: TokenSequenceLoader,
    activations_harvester: ActivationsHarvester,
) -> Iterator[ActivationsWithText]:
    for seq in token_sequence_loader.get_sequences_batch_iterator():
        activations_HSMPD = activations_harvester.get_activations_HSMPD(seq.tokens_HS)
        assert activations_HSMPD.shape[3] == 1
        yield ActivationsWithText(
            activations_HbSXD=activations_HSMPD[:, :, :, 0].to(device),
            tokens_HbS=seq.tokens_HS.to(device),
        )


examples_iterator = iterate_activations_with_text(
    token_sequence_loader=token_sequence_loader,
    activations_harvester=activations_harvester,
)
# %%

exam = LatentExaminer(cc=cc, activations_with_text_iter=examples_iterator)

# %%

# get the 1000 lowest and highest decoder norm indices
high_norms, high_indices = relative_decoder_norms_H.topk(4000, dim=-1, sorted=True)
low_norms, low_indices = relative_decoder_norms_H.topk(4000, dim=-1, largest=False, sorted=True)
latents_to_inspect_Hl = torch.cat([low_indices, high_indices])

# %%

high_indices = (relative_decoder_norms_H > 0.8).nonzero()[:, 0]
relative_decoder_norms_H[high_indices]
# %%
low_indices = (relative_decoder_norms_H < 0.2).nonzero()[:, 0]
relative_decoder_norms_H[low_indices]
# %%
latents_to_inspect_Hl = torch.cat([low_indices, high_indices])
latents_to_inspect_Hl.shape
# %%

latent_summaries = exam.gather_latent_summaries(
    total_batches=3000,
    latents_to_inspect_Hl=latents_to_inspect_Hl,
    topk_per_latent_per_batch=None,
)

# %%

print(len(latent_summaries))
print([(len(s.selected_examples), s.index, relative_decoder_norms_H[s.index].item()) for s in latent_summaries])

# %%

for summary in latent_summaries:
    topk_seqs_table(
        summary,
        tokenizer,
        extra_detail=f"relative decoder norm: {relative_decoder_norms_H[summary.index].item():.3f}",
        topk=50,
        context_size=1000,
    )

# %%


def generate_with_steering(
    llm: HookedTransformer,
    prompt: str,
    max_new_tokens: int,
    hookpoint: str,
    steering_coefficient: float,
    latent_idx: int,
) -> str:
    def steering_hook(activations: Tensor, hook: HookPoint) -> Tensor:
        return activations + steering_coefficient * sae_W_dec_HD[latent_idx]

    with llm.hooks(fwd_hooks=[(hookpoint, steering_hook)]):
        output = cast(
            str,
            llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                return_type="str",
                prepend_bos=False,  # handled by dataloader
            ),
        )

    return output


# %%

prompt = "When I look at myself in the mirror, I see"
latent_idx = 248

print(f"Prompt: {prompt}")
no_steering_output = cast(str, llms[0].generate(prompt, max_new_tokens=50))  # type: ignore

table = Table(show_header=False, show_lines=True, title="Steering Output")
table.add_row("Normal", no_steering_output)
for i in tqdm(range(3), "Generating steered examples..."):
    table.add_row(
        f"Steered #{i}",
        generate_with_steering(
            llm=llms[0],  # type: ignore
            prompt=prompt,
            max_new_tokens=20,
            hookpoint="blocks.8.hook_resid_post",
            steering_coefficient=20.0,  # roughly 1.5-2x the latent's max activation
            latent_idx=latent_idx,
        ).replace("\n", "↵"),
    )
rprint(table)

# %%
