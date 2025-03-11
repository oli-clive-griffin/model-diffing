# %%
import os
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm import tqdm  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformer_lens.hook_points import HookPoint  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.token_loader import HuggingfaceTextDatasetConfig, build_tokens_sequence_loader
from model_diffing.interp import (
    LatentExample,
    LatentSummary,
    display_top_seqs,
    gather_max_activating_examples,
    iterate_activations_with_text,
    top_and_bottom_logits,
)
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateExperimentConfig
from model_diffing.utils import get_device, not_none

# %% Setup

print(f"Current working directory: {Path.cwd()}")
os.chdir(Path(__file__).parent.parent.parent)
print(f"Current working directory: {Path.cwd()}")

device = get_device()
cache_dir = ".cache"

DOWNLOAD_DIR = ".data/artifact_download"
# download_experiment_checkpoint(
#     # this run is a layer-8 jumprelu SAE on pythia-160m for ≈ 90m tokens
#     # tokens = (checkpoint_idx * checkpoint_every_n_batches * batch_size)
#     #        = (12 * 1000 * 8192)
#     run_id="tu1pvnl6",
#     version="v12",
#     destination_dir=DOWNLOAD_DIR,
# )

sae: AcausalCrosscoder[Any] = AcausalCrosscoder.load(Path(DOWNLOAD_DIR) / "model")
assert sae.is_folded.item()

with open(Path(DOWNLOAD_DIR) / "experiment_config.yaml") as f:
    exp_config = JanUpdateExperimentConfig(**yaml.safe_load(f))

assert len(exp_config.data.activations_harvester.llms) == 1
llm_cfg = exp_config.data.activations_harvester.llms[0]
llm = HookedTransformer.from_pretrained(not_none(llm_cfg.name), revision=llm_cfg.revision)
llm = cast(HookedTransformer, llm.to(device))
tokenizer = llm.tokenizer
if not isinstance(tokenizer, PreTrainedTokenizerBase):
    raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

# without this, it'll be really slow to run locally
assert isinstance(exp_config.data.token_sequence_loader, HuggingfaceTextDatasetConfig)
exp_config.data.token_sequence_loader.shuffle_buffer_size = 32  # type: ignore

# first, get an iterator over sequences of tokens
batch_size = 1  # todo rethink me
sequence_length = exp_config.data.token_sequence_loader.sequence_length  # type: ignore

token_sequence_loader = build_tokens_sequence_loader(
    cfg=exp_config.data.token_sequence_loader,
    cache_dir=cache_dir,
    tokenizer=tokenizer,
    batch_size=batch_size,
)

activations_harvester = ActivationsHarvester(
    llms=[llm],
    hookpoints=exp_config.hookpoints,
)

examples_iterator = iterate_activations_with_text(
    token_sequence_loader=token_sequence_loader,
    activations_harvester=activations_harvester,
)

# %% Interp

# max-activating-examples

latents_to_inspect = list(range(100))

examples_by_latent = gather_max_activating_examples(
    examples_iterator,
    cc=sae,
    total_batches=20,
    context_size=20,
    topk_per_latent_per_batch=5,
    latents_to_inspect=latents_to_inspect,
)

# %%


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
    top_and_bottom_logits(
        llm,
        sae_W_dec_HD=sae_W_dec_HD,
        latent_indices=[latent_index],
    )


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
no_steering_output = cast(str, llm.generate(prompt, max_new_tokens=50))

table = Table(show_header=False, show_lines=True, title="Steering Output")
table.add_row("Normal", no_steering_output)
for i in tqdm(range(3), "Generating steered examples..."):
    table.add_row(
        f"Steered #{i}",
        generate_with_steering(
            llm,
            prompt,
            max_new_tokens=20,
            hookpoint="blocks.8.hook_resid_post",
            steering_coefficient=20.0,  # roughly 1.5-2x the latent's max activation
            latent_idx=latent_idx,
        ).replace("\n", "↵"),
    )
rprint(table)

# %%
