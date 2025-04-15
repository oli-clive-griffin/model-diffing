import random
from collections.abc import Iterator
from itertools import islice
from typing import TypeVar

import torch
from tqdm import tqdm

# Shapes:
# B = "batch"
# X = "arbitrary shape"


def batch_shuffle_tensor_iterator_BX(
    tensor_iterator_X: Iterator[torch.Tensor],  # X here means "arbitrary shape"
    yield_batch_size_B: int,
    shuffle_buffer_size: int,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    batches_iter = _batch_shuffle_iterator(
        iterator=tensor_iterator_X,
        yield_batch_size=yield_batch_size_B,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )
    return map(torch.stack, batches_iter)


T = TypeVar("T")


def _batch_shuffle_iterator(
    iterator: Iterator[T],
    yield_batch_size: int,
    shuffle_buffer_size: int,
    seed: int = 42,
) -> Iterator[list[T]]:
    rng = random.Random(seed)
    refill_threshold = yield_batch_size // 2

    if shuffle_buffer_size < refill_threshold:
        raise ValueError(f"{shuffle_buffer_size=} must be greater than {refill_threshold=}")

    buffer_BfX: list[T] = []

    def refill():
        needed = shuffle_buffer_size - len(buffer_BfX)
        for item in tqdm(islice(iterator, needed), total=needed, desc="refilling buffer"):
            buffer_BfX.append(item)
        rng.shuffle(buffer_BfX)

    def yield_batch():
        yield buffer_BfX[-yield_batch_size:]
        del buffer_BfX[-yield_batch_size:]

    while True:
        refill()

        # If the buffer wasn't refilled above, the iterator is exhausted, so we yield the remaining activations and break
        if len(buffer_BfX) <= shuffle_buffer_size // 2:
            while len(buffer_BfX) >= yield_batch_size:
                yield from yield_batch()
            break

        # yield batches until buffer is half empty
        while len(buffer_BfX) > shuffle_buffer_size // 2:
            yield from yield_batch()
