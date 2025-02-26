# %%
import asyncio
import concurrent.futures
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import Thread
from typing import cast

import anthropic
import torch
from datasets import DatasetDict, load_dataset
from pydantic import BaseModel
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

BASE = "Qwen/Qwen2.5-Math-1.5B"
R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

hf_math = AutoModelForCausalLM.from_pretrained(BASE, cache_dir=".cache")
hf_r1 = AutoModelForCausalLM.from_pretrained(R1, cache_dir=".cache")

# %%

# this should work because the model is the same, just a finetune
llm_math = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", hf_model=hf_math)
llm_r1 = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", hf_model=hf_r1)


# %%

tokenizer_r1 = AutoTokenizer.from_pretrained(R1)
tokenizer_base = AutoTokenizer.from_pretrained(BASE)

#

# TODO check tokenizers are the same

# %%
print(llm_math.generate("User: what's the square root of 16?", temperature=0))
print(llm_r1.generate("User: what's the square root of 16?", temperature=0))

# %%

ds: DatasetDict = load_dataset("HuggingFaceH4/MATH-500")  # type: ignore

# %%

MATH_ds_geom: DatasetDict = load_dataset("EleutherAI/hendrycks_math", "geometry")  # type: ignore
MATH_ds_alg: DatasetDict = load_dataset("EleutherAI/hendrycks_math", "algebra")  # type: ignore
MATH_ds_prob: DatasetDict = load_dataset("EleutherAI/hendrycks_math", "counting_and_probability")  # type: ignore

# %%

api = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class JudgeResponse(BaseModel):
    judge_response: str
    correct: bool | None = None


async def judge_async(problem: str, solution: str, answer: str, given_answer: str) -> JudgeResponse:
    prompt = f"""
    You are a math expert judging another's work. You are given a problem, the correct working and answer, and the answer given by the person you are judging.
    You need to determine if the given answer is correct.

    Problem:
    {problem}

    Correct Working:
    {solution}

    Correct Answer:
    {answer}

    Given Answer: (judge this)
    {given_answer}

    Please finish your response with "True" if the given answer is correct, and "False" if it is not.
    """
    JUDGE = "claude-3-5-haiku-latest"
    response = api.messages.create(
        model=JUDGE,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        stream=True,
    )
    generated_text = ""
    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            generated_text += event.delta.text
            print(event.delta.text, end="")
    if ("true" in generated_text.lower() and "false" in generated_text.lower()) or (
        "true" not in generated_text.lower() and "false" not in generated_text.lower()
    ):
        return JudgeResponse(
            judge_response=generated_text,
            correct=None,
        )
    return JudgeResponse(
        judge_response=generated_text,
        correct="true" in generated_text.lower(),
    )


# %%


def get_answer(problem: str, max_new_tokens: int = 1000, print_progress: bool = False) -> str:
    inputs = tokenizer_r1(problem, return_tensors="pt", padding=False, add_special_tokens=False)
    if print_progress:
        streamer = TextIteratorStreamer(tokenizer_r1, skip_prompt=True)  # type: ignore
        generation_kwargs = dict(inputs, max_new_tokens=max_new_tokens, streamer=streamer)
        thread = Thread(target=llm_r1.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="")
            generated_text += new_text

        return generated_text

    return llm_r1.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore


# %%
async def judge_bulk(examples: list[dict[str, str]], generated_answers: list[str]) -> tuple[list[JudgeResponse], float]:
    print("\nJudging all answers asynchronously...")
    judge_tasks: list[asyncio.Task[JudgeResponse]] = []
    # Create all judging tasks
    for ex, answer in zip(examples, generated_answers, strict=True):
        judge_task = asyncio.create_task(judge_async(ex["problem"], ex["solution"], ex["answer"], answer))
        judge_tasks.append(judge_task)

    # Wait for all judging tasks to complete
    judge_responses = await asyncio.gather(*judge_tasks, return_exceptions=False)

    pct_correct = sum(res.correct or 0 for res in judge_responses) / len(judge_responses)

    print(f"\nPercentage correct: {pct_correct:.2%}")

    return judge_responses, pct_correct


# %%

prompts = ds["test"]["problem"]
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    # Create a list of futures
    futures = [executor.submit(get_answer, prompt) for prompt in prompts]

    # Use tqdm to track the completion of futures
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        results.append(future.result())


# %%

print("hello")

# %%
ex = ds["test"][0]

print(f"""
Problem:
{ex["problem"]}

Solution:
{ex["solution"]}

Answer:
{ex["answer"]}
""")

answer = get_answer(ex["problem"], print_progress=True)

judge_response = judge_async(ex["problem"], ex["solution"], ex["answer"], answer)
