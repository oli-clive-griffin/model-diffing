from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import tabulate  # type: ignore
import torch
from einops import rearrange
from rich import print as rprint
from rich.table import Table
from tqdm import tqdm  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from crosscoding.log import logger
from crosscoding.models.sparse_coders import ModelHookpointAcausalCrosscoder
from crosscoding.utils import compute_relative_norms_N


@dataclass
class ActivationsWithText:
    activations_HSXD: torch.Tensor
    tokens_HS: torch.Tensor


@dataclass
class LatentExample:
    tokens_S: torch.Tensor
    last_tok_latent_act: float


@dataclass
class LatentSummary:
    index: int
    selected_examples: list[LatentExample]
    """sorted by descending activation strength"""
    density: float


class LatentExaminer:
    def __init__(
        self,
        cc: ModelHookpointAcausalCrosscoder[Any],
        activations_with_text_iter: Iterator[ActivationsWithText],
    ):
        self.cc_unit_normed = cc.with_decoder_unit_norm()
        self.activations_with_text_iter = activations_with_text_iter

    # TODO add method to run forward pass on only the relevant latents

    @torch.no_grad()
    def gather_latent_summaries(
        self,
        latents_to_inspect_L: torch.Tensor | list[int],
        total_batches: int = 200,
        separation_threshold_tokens: int = 5,
        topk_per_latent_per_batch: int | None = None,
    ) -> list[LatentSummary]:
        if isinstance(latents_to_inspect_L, list):
            latents_to_inspect_L = torch.tensor(latents_to_inspect_L)

        examples_by_latent_idx = defaultdict[int, list[LatentExample]](list)
        num_firings_by_latent_idx = defaultdict[int, int](int)

        total_firings = 0
        skipped_firings = 0
        not_skipped_firings = 0

        tokens_seen = 0

        pbar = tqdm(
            islice(self.activations_with_text_iter, total_batches),
            total=total_batches,
            desc="Gathering latent examples",
        )

        for activations_with_text in pbar:
            # TODO refactor this out
            latents_HSL = self.get_latents(activations_with_text)
            H, S, _L = latents_HSL.shape

            if topk_per_latent_per_batch is not None:
                # zero out the activations that are not in the topk
                vals, indices = latents_HSL.topk(topk_per_latent_per_batch, dim=-1)
                latents_HSL = torch.zeros_like(latents_HSL).scatter_(-1, indices, vals)

            for h_idx in range(H):
                latents_SL = latents_HSL[h_idx]
                tokens_S = activations_with_text.tokens_HS[h_idx]

                # remove sequences where our latents are not activated at all
                active_latents_mask_L = latents_SL[:, latents_to_inspect_L].sum(dim=0)
                active_latents_La = latents_to_inspect_L[active_latents_mask_L.nonzero(as_tuple=True)[0]]
                pct_latents_active = active_latents_La.numel() / latents_to_inspect_L.numel()
                pbar.set_postfix(
                    pct_latents_active=f"{pct_latents_active:.3%}",
                    num_firings=total_firings,
                    n_latents_fired=len(examples_by_latent_idx),
                )

                for latent_idx in active_latents_La.tolist():
                    latents_S = latents_SL[:, latent_idx]
                    (seq_pos_indices,) = latents_S.nonzero(as_tuple=True)
                    total_firings += seq_pos_indices.numel()
                    last_pos = -separation_threshold_tokens
                    for pos in seq_pos_indices:
                        # TODO <= vs <, not sure
                        if pos <= last_pos + separation_threshold_tokens:
                            skipped_firings += 1
                        else:
                            not_skipped_firings += 1
                            example = LatentExample(
                                tokens_S=tokens_S[: pos + 1],
                                last_tok_latent_act=latents_S[pos].item(),
                            )
                            examples_by_latent_idx[latent_idx].append(example)
                            last_pos = pos

                    num_firings_by_latent_idx[latent_idx] += seq_pos_indices.numel()

            tokens_seen += H * S

        logger.info(f"Skipped {skipped_firings} examples, {not_skipped_firings} examples not skipped")
        logger.info(f"Seen {tokens_seen} tokens")

        return [
            LatentSummary(
                index=latent_idx,
                selected_examples=sorted(examples, key=lambda x: x.last_tok_latent_act, reverse=True),
                density=num_firings_by_latent_idx[latent_idx] / (tokens_seen + 1e-6),
            )
            for latent_idx, examples in examples_by_latent_idx.items()
        ]

    def get_latents(self, activations_with_text: ActivationsWithText):
        H, S, *_ = activations_with_text.activations_HSXD.shape
        activations_HsXD = rearrange(activations_with_text.activations_HSXD, "h s ... -> (h s) ...")
        latents_HsL = self.cc_unit_normed.forward_train(activations_HsXD).latents_BL
        latents_HSL = rearrange(latents_HsL, "(h s) l -> h s l", h=H, s=S)
        return latents_HSL


# TODO this function isn't good enough to ship with incorrect logit lens impl
def top_and_bottom_logits(
    llm: HookedTransformer,
    sae_W_dec_LD: torch.Tensor,
    latent_indices: list[int] | None = None,
    k: int = 10,
):
    latent_indices_iter = latent_indices if latent_indices is not None else list(range(sae_W_dec_LD.shape[0]))

    for latent_idx in latent_indices_iter:
        # TODO use act cache and layernorm stuff here
        logits = sae_W_dec_LD[latent_idx] @ llm.W_U
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


def display_top_seqs(data: list[tuple[float, list[str]]], latent_index: int, density: float):
    """
    Given a list of (activation: float, str_toks: list[str]), displays a table of these sequences, with
    the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens (usually weird quotation marks) for readability.
    """
    table = Table(
        "Act",
        "Sequence",
        title=f"Max Activating Examples (latent {latent_index}, density={(density * 100):.4f}%)",
        show_lines=True,
    )
    for act, str_toks in data:
        str_toks[-1] = f"[b u green]{str_toks[-1]}[/]"
        formatted_seq = "".join(str_toks).replace("�", "").replace("\n", "↵")
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)


def topk_seqs_table(
    summary: LatentSummary,
    tokenizer: PreTrainedTokenizerBase,
    extra_detail: str,
    topk: int = 10,
    context_size: int = 20,
):
    """
    Given a list of (activation: float, str_toks: list[str]), displays a table of these sequences, with
    the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens (usually weird quotation marks) for readability.
    """
    table = Table(
        "Act",
        "Sequence",
        title=f"Max Activating Examples (latent {summary.index}, density={(summary.density * 100):.4f}%, {extra_detail})",
        show_lines=True,
    )

    for example in summary.selected_examples[:topk]:
        str_toks = [tokenizer.decode(tok) for tok in example.tokens_S[-context_size:]]
        str_toks[-1] = f"[b u green]{str_toks[-1]}[/]"
        formatted_seq = (
            "".join(str_toks)
            .replace("�", "")
            .replace("\n", "↵")
            .replace("<think>", "[b u orange_red1]<think>[/]")
            .replace("</think>", "[b u orange_red1]</think>[/]")
        )

        table.add_row(
            f"{example.last_tok_latent_act:.3f}",
            repr(formatted_seq),
        )

    rprint(table)
