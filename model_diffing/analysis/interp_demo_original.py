# %%
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
    LatentExample,
    LatentSummary,
    display_top_seqs,
    gather_max_activating_examples,
    top_and_bottom_logits,
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

DOWNLOAD_DIR = ".data/artifact_download"
download_experiment_checkpoint(
    # this run is a layer-8 jumprelu SAE on pythia-160m for ≈ 90m tokens
    # tokens = (checkpoint_idx * checkpoint_every_n_batches * batch_size)
    #        = (12 * 1000 * 8192)
    run_id="ar86synr",
    version="v1",
    destination_dir=DOWNLOAD_DIR,
)

# %%
sae: AcausalCrosscoder[Any] = AcausalCrosscoder.load(Path(DOWNLOAD_DIR) / "model")
assert sae.is_folded.item()
sae.to(device)

with open(Path(DOWNLOAD_DIR) / "experiment_config.yaml") as f:
    exp_config = JumpReLUModelDiffingFebUpdateExperimentConfig(**yaml.safe_load(f))

# assert len(exp_config.data.activations_harvester.llms) == 1

llms = build_llms(
    exp_config.data.activations_harvester.llms,
    cache_dir=cache_dir,
    device=device,
    inferenced_type="bfloat16",
)

assert isinstance(exp_config.data.token_sequence_loader, MathDatasetConfig)

# %%

# first, get an iterator over sequences of tokens
batch_size = 2  # todo rethink me
sequence_length = exp_config.data.token_sequence_loader.max_sequence_length  # type: ignore

token_sequence_loader = build_tokens_sequence_loader(
    cfg=exp_config.data.token_sequence_loader,
    cache_dir=cache_dir,
    tokenizer=llms[0].tokenizer,  # type: ignore
    batch_size=batch_size,
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
            activations_HSMD=activations_HSMPD[:, :, :, 0].to(device),
            tokens_HS=seq.tokens_HS.to(device),
        )

examples_iterator = iterate_activations_with_text(
    token_sequence_loader=token_sequence_loader,
    activations_harvester=activations_harvester,
)
# %%

# max-activating-examples

latents_to_inspect = list(range(10))

examples_by_latent = gather_max_activating_examples(
    examples_iterator,
    cc=sae,
    total_batches=2,
    context_size=20,
    topk_per_latent_per_batch=5,
    latents_to_inspect=latents_to_inspect,
)

# %%

tokenizer = llms[0].tokenizer  # type: ignore

def as_table_data(ex: LatentExample) -> tuple[float, list[str]]:
    tokens_strings = [tokenizer.decode(tok) for tok in ex.tokens_S]  # type: ignore
    return (ex.last_tok_hidden_act, tokens_strings)


# display up to 10 examples for each latent
sae_W_dec_HD = sae.W_dec_HXD[:, 0, 0, :]


def examine_latent(latent_index: int, summary: LatentSummary):
    display_top_seqs(
        data=[as_table_data(example) for example in summary.selected_examples[:20]],
        latent_index=latent_index,
        density=summary.density,
    )
    # top_and_bottom_logits(
    #     llm,
    #     sae_W_dec_HD=sae_W_dec_HD,
    #     latent_indices=[latent_index],
    # )


for latent_index, summary in examples_by_latent.items():
    examine_latent(latent_index, summary)
    print("\n=========")

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
