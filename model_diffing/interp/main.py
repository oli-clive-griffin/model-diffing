from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import tabulate
import torch
from einops import rearrange
from rich.table import Table
from rich import print as rprint
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.data.model_hookpoint_dataloader import ScaledModelHookpointActivationsDataloader
from model_diffing.models.crosscoder import AcausalCrosscoder


@dataclass
class LatentExample:
    tokens_context_S: torch.Tensor
    latents_context_S: torch.Tensor


@torch.no_grad()
def gather_max_activations(
    dataloader: ScaledModelHookpointActivationsDataloader,
    cc: AcausalCrosscoder[Any],
    total_batches: int = 200,
    context_size: int = 20,
    topk_per_latent: int = 50,
) -> list[list[LatentExample]]:
    """
    iterate over the dataloader, gathering context for all non-zero latents, in terms of tokens and latent activations

    returns:
        list of lists of LatentExample, where the outer list is over latents, and the inner list is over activating examples
    """
    latent_examples: list[list[LatentExample]] = [[] for _ in range(cc.hidden_dim)]

    for activations_with_text in islice(dataloader.iterate_activations_with_text(), total_batches):
        B, S = activations_with_text.activations_BSMPD.shape[:2]
        activations_BsMPD = rearrange(activations_with_text.activations_BSMPD, "b s m p d -> (b s) m p d")
        res = cc.forward_train(activations_BsMPD)
        hidden_BSH = rearrange(res.hidden_BH, "(b s) h -> b s h", b=B, s=S)

        # zero out the activations that are not in the topk
        vals, indices = hidden_BSH.topk(topk_per_latent, dim=-1)
        hidden_BSH = torch.zeros_like(hidden_BSH).scatter_(-1, indices, vals)

        for latent_idx in range(hidden_BSH.shape[-1]):
            hidden_BS = hidden_BSH[:, :, latent_idx]
            this_latent_examples = collect_examples(hidden_BS, activations_with_text.tokens_BS, context_size)
            latent_examples[latent_idx].extend(this_latent_examples)

    # within each latent, sort by activation strength.
    for latent_idx in range(cc.hidden_dim):
        latent_examples[latent_idx].sort(key=lambda x: x.latents_context_S[-1].item())

    return latent_examples


def collect_examples(latents_BS: torch.Tensor, tokens_BS: torch.Tensor, context_size: int) -> Iterator[LatentExample]:
    batch_indices, seq_pos_indices = latents_BS.nonzero(as_tuple=True)
    seq_pos_start_indices = (seq_pos_indices - context_size).clamp(min=0)
    for firing_batch, firing_seq_pos_start, firing_seq_pos_end in zip(
        batch_indices, seq_pos_start_indices, seq_pos_indices, strict=False
    ):
        yield LatentExample(
            tokens_context_S=tokens_BS[firing_batch, firing_seq_pos_start:firing_seq_pos_end],
            latents_context_S=latents_BS[firing_batch, firing_seq_pos_start:firing_seq_pos_end],
        )


def display_latent_examples(latent_examples: list[LatentExample], tokenizer: PreTrainedTokenizerBase):
    for i, example in enumerate(latent_examples):
        print(f"Example {i}")
        print(tokenizer.decode(example.tokens_context_S))
        print(example.latents_context_S)
        print()


def top_and_bottom_logits(
    llm: HookedTransformer,
    sae_W_dec_HD: torch.Tensor,
    k: int = 10,
):
    for latent_idx in range(sae_W_dec_HD.shape[0]):
        logits = sae_W_dec_HD[latent_idx] @ llm.W_U
        pos_logits, pos_token_ids = logits.topk(k)
        pos_tokens = llm.to_str_tokens(pos_token_ids)
        neg_logits, neg_token_ids = logits.topk(k, largest=False)
        neg_tokens = llm.to_str_tokens(neg_token_ids)

        print(
            tabulate.tabulate(
                zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits, strict=True),
                headers=["Bottom tokens", "Value", "Top tokens", "Value"],
                tablefmt="simple_outline",
                stralign="right",
                numalign="left",
                floatfmt="+.3f",
            )
        )


def display_top_seqs(data: list[tuple[float, list[str]]]):
    """
    Given a list of (activation: float, str_toks: list[str]), displays a table of these sequences, with
    the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens � (usually weird quotation marks) for readability.
    """
    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks in data:
        formatted_seq = (
            "".join(
                [
                    f"[b u green]{str_tok}[/]" if i == len(str_toks) - 1 else str_tok
                    for i, str_tok in enumerate(str_toks)
                ]
            )
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)

    # DEMO_data = [
    #     (1.0, ["Hello", " world", "!"]),
    #     (0.5, ["Hello", " there", "!"]),
    #     (0.0, ["Hello", "!", " there"]),
    # ]

    # display_top_seqs(DEMO_data)
