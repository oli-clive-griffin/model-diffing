# %%

from collections import defaultdict
from datetime import datetime
from itertools import islice
import sys
import textwrap
from typing import cast

import torch
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.model_hookpoint_dataloader import ScaledModelHookpointActivationsDataloader
from model_diffing.data.token_loader import MathDatasetTokenSequenceLoader
from model_diffing.utils import get_device, inspect

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

# llm_r1.tokenizer.padding_side
# # %%
# llm_r1.tokenizer.special_tokens_map

# # %%
# llm_math.tokenizer.pad_token
# %%


# %%

tokenizer_base = AutoTokenizer.from_pretrained(BASE)
tokenizer_r1 = AutoTokenizer.from_pretrained(R1)

# # %%


def enc_str(s: str) -> torch.Tensor:
    return cast(torch.Tensor, tokenizer_r1.encode(s, return_tensors="pt", add_special_tokens=False))[0]


# %%
device = get_device()

# %%

# toks_S = first.tokens_BS[0][~first.special_tokens_mask_BS[0]].to(device)
# inspect(toks_S)
# # %%
# tokenizer_r1.decode(toks_S)

# %%
THINK = torch.tensor(151648).to(device)
END_THINK = torch.tensor(151649).to(device)
THINK_1 = THINK[None, ...]
END_THINK_1 = END_THINK[None, ...]
# %%


def generate_S(
    llm: HookedTransformer,
    question: str,
    suffix_toks_S: torch.Tensor | None = None,
    print_prompt: bool = False,
    **generate_kwargs,
):
    batch_1S = cast(torch.Tensor, tokenizer_r1.encode(question, return_tensors="pt")).to(device)
    assert batch_1S.shape[0] == 1

    if suffix_toks_S is not None:
        suffix_toks_S = suffix_toks_S.to(device)
        assert suffix_toks_S.ndim == 1
        batch_1S = torch.cat([batch_1S, suffix_toks_S[None, ...]], dim=1)

    input_len = batch_1S.shape[1]

    if print_prompt:
        print(f"runnning prompt:\n{tokenizer_r1.decode(batch_1S[0])}")

    out_BS = cast(
        torch.Tensor,
        llm.generate(
            batch_1S,
            prepend_bos=True,
            return_type="tokens",
            **generate_kwargs,
        ),
    )
    assert out_BS.shape[0] == 1
    return out_BS[0], input_len


# base_question = "What is 123 * 234"

# %%

# output_S = generate_r1_S(base_question, THINK_1)

# # %%

# output_str = tokenizer_r1.decode(output_S)
# print(textwrap.fill(output_str, width=100))

# # %%

# output_no_think_S = generate_r1_S(
#     base_question,
#     tokenizer_r1.encode(
#         "<think></think>The answer is \\(\\boxed{",
#         return_tensors="pt",
#         add_special_tokens=False,
#     )[0],
#     max_new_tokens=10,
# )

# %%

# output_no_think_str = tokenizer_r1.decode(output_no_think_S)
# print(textwrap.fill(output_no_think_str, width=100))

base_question = "What is %s * %s?"
pairs = [(34, 74), (562, 34), (83, 22)]

n_elipses_options = [1, 2, 3, 4, 6, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50]
n_samples_each = 20

elipses_S = cast(torch.Tensor, tokenizer_r1.encode("...", return_tensors="pt", add_special_tokens=False)).to(device)[0]

# %%

force_answer = "The answer is \\(\\boxed{"
force_answer_S = cast(
    torch.Tensor, tokenizer_r1.encode(force_answer, return_tensors="pt", add_special_tokens=False)
).to(device)[0]
# %%


def run_token_filler_experiment_multiple(
    llm: HookedTransformer,
    pairs: list[tuple[int, int]],
    n_fillers_options: list[int],
    n_samples_each: int,
    filler_token_idx: int,
):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def log(s: str):
        with open(f"thinking_log_{now}.txt", "a") as f:
            f.write(s + "\n")

    results: dict[
        tuple[int, int],  # which operands
        dict[
            int,  # n_elipses
            float,  # pct_correct
        ],
    ] = {(a, b): {} for a, b in pairs}

    for a, b in tqdm(pairs, position=0, leave=True):
        question = base_question % (a, b)
        log(f"\n\n\n# RUNNING |{question}|\n")
        q_results: dict[int, float] = {}
        for n_filler_toks in tqdm(n_fillers_options, position=1, leave=False):
            n_correct = 0
            log(f"\n\nn_filler_tokens: {n_filler_toks}")
            for _ in tqdm(range(n_samples_each), position=2, leave=False):
                output, input_len = generate_S(
                    llm,
                    question,
                    torch.cat(
                        [
                            THINK_1,
                            torch.tensor([filler_token_idx], device=device).repeat(n_filler_toks),
                            END_THINK_1,
                            force_answer_S,
                        ]
                    ),
                    False,
                    max_new_tokens=8,
                    verbose=False,
                )

                log(f"output: |{tokenizer_r1.decode(output[input_len:])}|")
                str_output = tokenizer_r1.decode(output[input_len:])

                num_str = ""
                for char in str_output:
                    if char == ",":
                        break
                    if char.isdigit() or char == ".":
                        num_str += char
                    else:
                        break
                if num_str == "":
                    continue

                try:
                    num = int(num_str)
                except ValueError:
                    continue

                correct = num == a * b
                log(f"correct={correct}")

                n_correct += correct

            pct_correct = n_correct / n_samples_each
            q_results[n_filler_toks] = pct_correct
            print(f"{n_filler_toks=}: {pct_correct=}")
        results[(a, b)] = q_results
    return results


# %%
# %%

# plots:


def visualize_results(
    results: list[
        tuple[
            str,  # model name
            int,  # filler token idx
            dict[tuple[int, int], dict[int, float]],  # results
        ]
    ],
):
    import matplotlib.pyplot as plt

    plt.close()

    for model_name, filler_token_idx, model_results in results:
        token = tokenizer_r1.decode([filler_token_idx]).replace("\n", "↵")
        for (a, b), q_results in model_results.items():
            label = f"{model_name} | {base_question % (a, b)} | filter_token={token}"
            x = list(q_results.keys())
            y = list(q_results.values())
            plt.plot(
                x,
                y,
                label=label,
                marker="o",
                markersize=4,
                linestyle="",
            )

    plt.title("zero shot accuracy with elipsis-only thinking")
    plt.xlabel("n_elipses")
    plt.ylabel("pct of times correct answer given")
    plt.legend()
    plt.show()

    # q_S = tokenizer_r1.encode(base_question % (34, 74), return_tensors="pt").to(device)[0]
    # manyelipses_S = (
    #     cast(torch.Tensor, tokenizer_r1.encode("...", return_tensors="pt", add_special_tokens=False))
    #     .to(device)[0]
    #     .repeat(7)
    # )
    # example = tokenizer_r1.decode(torch.cat([q_S, THINK_1, manyelipses_S, END_THINK_1, force_answer_S]))
    # print(f"example prompt:\n{example}")


# %%

visualize_results(results_shared, "r1")  # type: ignore
# %%

elip_toks = enc_str("...")
assert elip_toks.shape == (1,)
ellipis_idx = cast(int, elip_toks[0].item())

# %%

results_focused = run_token_filler_experiment_multiple(
    llm_r1,
    pairs=[(83, 22)],
    n_fillers_options=list(range(50)),
    n_samples_each=20,
    filler_token_idx=ellipis_idx,
)
# %%
# visualize_results(results_focused, "r1")
# %%
results_focused_math = run_token_filler_experiment_multiple(
    llm_math,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=20,
    filler_token_idx=ellipis_idx,
)
# %%
del results_focused_math[(34, 74)]
del results_focused_math[(562, 34)]
visualize_results([("math", results_focused_math), ("r1", results_focused)])
# %%

toks = cast(
    torch.Tensor,
    tokenizer_r1.encode("\n\n", return_tensors="pt", add_special_tokens=False),
)[0]
assert toks.shape == (1,)

return_tok_idx = cast(int, toks[0].item())
# %%

r1_ret_res = run_token_filler_experiment_multiple(
    llm_r1,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=return_tok_idx,
)

math_ret_res = run_token_filler_experiment_multiple(
    llm_math,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=return_tok_idx,
)

r1_elip_res = run_token_filler_experiment_multiple(
    llm_r1,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=ellipis_idx,
)

math_elip_res = run_token_filler_experiment_multiple(
    llm_math,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=ellipis_idx,
)

r1_rand_res = run_token_filler_experiment_multiple(
    llm_r1,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=123,
)

math_rand_res = run_token_filler_experiment_multiple(
    llm_math,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=123,
)
# %%
# %%
for results in [
    ("r1", ellipis_idx, r1_elip_res),
    ("math", ellipis_idx, math_elip_res),
    ("r1", return_tok_idx, r1_ret_res),
    ("math", return_tok_idx, math_ret_res),
    ("r1", 123, r1_rand_res),
    ("math", 123, math_rand_res),
]:
    visualize_results([results])
# %%

toks = cast(torch.Tensor, tokenizer_r1.encode("hm", return_tensors="pt", add_special_tokens=False))
assert toks.shape == (1, 1), toks.shape
hmm_idx = cast(int, toks[0][0].item())

r1_hmm_res = run_token_filler_experiment_multiple(
    llm_r1,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=hmm_idx,
)

math_hmm_res = run_token_filler_experiment_multiple(
    llm_math,
    pairs=[(83, 22)],
    n_fillers_options=list(range(0, 50, 5)),
    n_samples_each=10,
    filler_token_idx=hmm_idx,
)

# %%
visualize_results([("r1", hmm_idx, r1_hmm_res), ("math", hmm_idx, math_hmm_res)])

# %%
r1_hmm_res_long = run_token_filler_experiment_multiple(
    llm_r1,
    pairs=[(83, 22)],
    n_fillers_options=[1, 3, 6, 12, 20, 30, 42, 60, 80, 100],
    n_samples_each=20,
    filler_token_idx=hmm_idx,
)

math_hmm_res_long = run_token_filler_experiment_multiple(
    llm_math,
    pairs=[(83, 22)],
    n_fillers_options=[1, 3, 6, 12, 20, 30, 42, 60, 80, 100],
    n_samples_each=20,
    filler_token_idx=hmm_idx,
)

visualize_results([("r1", hmm_idx, r1_hmm_res_long), ("math", hmm_idx, math_hmm_res_long)])

# %%

r1_hmm_res_longer = run_token_filler_experiment_multiple(
    llm_r1,
    pairs=[(83, 22)],
    n_fillers_options=[1, 5, 30, 60, 100, 150],
    n_samples_each=20,
    filler_token_idx=hmm_idx,
)

math_hmm_res_longer = run_token_filler_experiment_multiple(
    llm_math,
    pairs=[(83, 22)],
    n_fillers_options=[1, 5, 30, 60, 100, 150],
    n_samples_each=20,
    filler_token_idx=hmm_idx,
)

visualize_results([("r1", hmm_idx, r1_hmm_res_longer), ("math", hmm_idx, math_hmm_res_longer)])
# %%

# def gen_thinking(llm: HookedTransformer, q_toks: torch.Tensor) -> torch.Tensor:
#     out_S = cast(
#         torch.Tensor,
#         llm.generate(
#             torch.cat([q_toks, THINK_1])[None, ...],
#             return_type="tokens",
#             max_new_tokens=1000,
#         ),
#     )[0]
#     return out_S

# %%


# str_toks = [tokenizer_r1.decode(tok, return_tensors="pt") for tok in out_S]
# all_used_toks = set(str_toks)
tok_indices_to_split_on = {
    tokenizer_r1.encode("\n\n", return_tensors="pt", add_special_tokens=False).item(),  # type: ignore
    tokenizer_r1.encode(".\n\n", return_tensors="pt", add_special_tokens=False).item(),  # type: ignore
    tokenizer_r1.encode("]\n\n", return_tensors="pt", add_special_tokens=False).item(),  # type: ignore
}

# for tok in all_used_toks:
#     tok_idx = tokenizer_r1.encode(tok, return_tensors="pt", add_special_tokens=False).item()  # type: ignore
#     if tok == "\n\n":
#         # print(tok.replace("\n", "↵"))
#         tok_indices_to_split_on.add(tok_idx)
# print(tok_indices_to_split_on)


def create_thinking_prefixes(context_S: torch.Tensor) -> list[torch.Tensor]:
    prefixes = []
    for i, tok_idx in enumerate(context_S.tolist()):
        if tok_idx in tok_indices_to_split_on:
            prefixes.append(context_S[:i])

    prefixes.append(context_S)

    return prefixes


# %%
from datasets import load_dataset  # type: ignore
from datasets import IterableDataset  # type: ignore

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
        cache_dir=".cache",
    ),
)


# %%
ds_iter = iter(ds)
# %%


def iter_thinking_prefixes():
    for row in ds_iter:
        conv = row["reannotated_messages"]
        user_message, assistant_message = conv[0]["content"], conv[1]["content"]
        user_toks = enc_str(user_message).to(device)
        assistant_thinking = assistant_message.split("</think>")[0]
        thinking_toks = enc_str(assistant_thinking).to(device)
        prefixes = create_thinking_prefixes(thinking_toks)
        # print("\n\n==================||||||==================")
        # print(tokenizer_r1.decode(thinking_toks))
        # print("==================|toks|==================")
        # print('|'.join([
        #     tokenizer_r1.decode(tok).replace("�", "").replace("\n", "↵")
        #     for tok in thinking_toks
        # ]))
        # # print("==================|prefixes|==================")
        yield user_toks, prefixes
        # print(tokenizer_r1.decode(prefix))
        # print("-----------------------------")


all_results: list[dict[int, float]] = []
for i, (user_toks_S, prefixes_toks) in enumerate(islice(iter_thinking_prefixes(), 20)):
    print(f"\n\nExample {i}\n=========================================\n")
    print(f"<user_question>\n{textwrap.fill(tokenizer_r1.decode(user_toks_S), width=100)}\n</user_question>")
    full_reasoning = torch.cat(
        [
            user_toks_S,
            prefixes_toks[-1],
            END_THINK_1,
            force_answer_S,
        ]
    )
    answer_logits_SL = llm_r1.forward(
        full_reasoning,
        return_type="logits",
    )[0]

    psuedocorrect_tok_idx = answer_logits_SL[-1, :].argmax()
    logit = answer_logits_SL[-1].softmax(dim=-1)[psuedocorrect_tok_idx]
    del answer_logits_SL  # make sure we don't use it, it's got the wrong suffix

    psuedocorrect_tok = tokenizer_r1.decode([psuedocorrect_tok_idx])
    print(f"\npsuedo-correct token: {psuedocorrect_tok}, (with prob: {logit.item():.2f})")

    ex_res = {}

    for prefix_tok_S in prefixes_toks:
        ctx_S = torch.cat([user_toks_S, prefix_tok_S, END_THINK_1, force_answer_S])
        generation_S = llm_r1.generate(
            ctx_S[None, ...],
            return_type="tokens",
            max_new_tokens=5,
            verbose=False,
            prepend_bos=True,
        )[0][1:]  # to remove the BOS token

        # # todo kv cache
        # logits_SL = llm_r1.forward(
        #     ctx_S[None, ...],
        #     return_type="logits",
        # )[0]

        res, act_cache = llm_r1.run_with_cache(ctx_S[None, ...])
        logits_SL = res.logits
        # decomp = act_cache.get_full_resid_decomposition(
        #     expand_neurons=False,
        #     return_labels=True,
        # )
        # act_cache.apply_ln_to_stack

        pred_tok_idx = logits_SL[-1].argmax()
        pred_tok = tokenizer_r1.decode([pred_tok_idx])
        logits_L = logits_SL[-1].softmax(dim=0)

        # answer = tokenizer_r1.decode(generation_S[ctx_S.shape[0] - 1 :])
        psuedo_correct = pred_tok == psuedocorrect_tok
        print(f"\n-- {len(prefix_tok_S)} thinking tokens -----------------------------")
        confidence = logits_L[psuedocorrect_tok_idx].item()
        print(f"predicted token: |{psuedocorrect_tok}| {'✅' if psuedo_correct else '❌'} ")
        print(
            f"logit of correct token: {confidence:.4f}. pred_idx: {pred_tok_idx.item()}, correct_idx: {psuedocorrect_tok_idx.item()}"
        )
        # print(f"  from context: {tokenizer_r1.decode(ctx_S)[-80:]}".replace("\n", "↵"))
        ex_res[len(prefix_tok_S)] = confidence
    all_results.append(ex_res)

# %%

# log the results as a bunch of lines on a plot

import matplotlib.pyplot as plt

for res in all_results:
    print(list(res.keys())[-1])
    x_max = list(res.keys())[-1]
    x = [xi / x_max for xi in res.keys()]
    y = list(res.values())
    plt.plot(
        x,
        y,
        marker="o",
        markersize=4,
        linestyle="",
    )

plt.title("Correct-Token confidence by CoT truncation")
plt.xlabel("CoT truncation")
plt.ylabel("confidence in correct answer")
plt.legend()
plt.show()

# %%

# from transformer_lens.utils import
# llm_r1.ln

# with torch.no_grad():
#     embed = llm_r1.generate(
#         "1 + 1 =",
#         return_type="embeds",
#         max_new_tokens=2,
#         verbose=False,
#         prepend_bos=True,
#     )[0]

#     # print(tokenizer_r1.decode([(embed @ llm_r1.W_U).argmax()]))
#     out = embed @ llm_r1.W_U
#     print(out.shape)
#     print(tokenizer_r1.decode(out.argmax(dim=-1)))

#     out = embed[-1] @ llm_r1.W_U
#     print(out.shape)
#     print(tokenizer_r1.decode(out.argmax()))
