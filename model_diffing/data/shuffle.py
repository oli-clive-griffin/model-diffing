import random
from collections.abc import Iterator
from typing import TypeVar

from sympy import itermonomials
import torch

from model_diffing.log import logger
from model_diffing.utils import size_human_readable

# Shapes:
# B = "batch"
# Bf = "buffer size"
# X = "arbitrary shape"


def batch_shuffle_tensor_iterator_BX(
    tensor_iterator_X: Iterator[torch.Tensor],  # X here means "arbitrary shape"
    shuffle_buffer_size: int,
    yield_batch_size: int,
    name: str | None = None,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    rng = random.Random(seed)

    if shuffle_buffer_size < yield_batch_size:
        raise ValueError(
            f"shuffle_buffer_size ({shuffle_buffer_size}) must be greater than yield_batch_size ({yield_batch_size})"
        )
    if shuffle_buffer_size < yield_batch_size * 4:
        logger.warning(
            f"shuffle_buffer_size ({shuffle_buffer_size}) is less than 4x yield_batch_size ({yield_batch_size}), this may lead to poor shuffling"
        )

    first_tensor_X = next(tensor_iterator_X)  # this "wastes" an example. This is ok.

    buffer_BfX = torch.empty(
        (shuffle_buffer_size, *first_tensor_X.shape),
        device=first_tensor_X.device,
        dtype=first_tensor_X.dtype,
    )
    logger.info(f"shuffle buffer size: {size_human_readable(buffer_BfX)}{f' ({name})' if name else ''}")

    buffer_BfX[0] = first_tensor_X
    available_indices = {0}
    stale_indices = set(range(1, shuffle_buffer_size))

    def sample_BX():
        batch_indices = rng.sample(list(available_indices), yield_batch_size)
        available_indices.difference_update(batch_indices)
        stale_indices.update(batch_indices)
        return buffer_BfX[batch_indices]

    while True:
        # refill buffer
        for stale_idx, example_X in zip(list(stale_indices), tensor_iterator_X, strict=False):
            buffer_BfX[stale_idx] = example_X
            available_indices.add(stale_idx)
            stale_indices.remove(stale_idx)

        # If the buffer wasn't refilled above, the iterator is exhausted, so we yield the remaining activations and break
        if len(available_indices) <= shuffle_buffer_size // 2:
            while len(available_indices) >= yield_batch_size:
                yield sample_BX()
            break

        # yield batches until buffer is half empty
        while len(available_indices) > shuffle_buffer_size // 2:
            yield sample_BX()


def batch_shuffle_tensor_iterator_BX_list(
    tensor_iterator_X: Iterator[torch.Tensor],  # X here means "arbitrary shape"
    shuffle_buffer_size: int,
    yield_batch_size: int,
    name: str | None = None,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    rng = random.Random(seed)

    if shuffle_buffer_size < yield_batch_size:
        raise ValueError(
            f"shuffle_buffer_size ({shuffle_buffer_size}) must be greater than yield_batch_size ({yield_batch_size})"
        )
    if shuffle_buffer_size < yield_batch_size * 4:
        logger.warning(
            f"shuffle_buffer_size ({shuffle_buffer_size}) is less than 4x yield_batch_size ({yield_batch_size}), this may lead to poor shuffling"
        )

    buffer_BfX: list[torch.Tensor | None] = [None] * shuffle_buffer_size
    available_indices = set()
    stale_indices = set(range(shuffle_buffer_size))

    def sample_BX():
        batch_indices = rng.sample(list(available_indices), yield_batch_size)
        available_indices.difference_update(batch_indices)
        stale_indices.update(batch_indices)
        return torch.stack([buffer_BfX[i] for i in batch_indices])  # type: ignore

    while True:
        # refill buffer
        for stale_idx, example_X in zip(list(stale_indices), tensor_iterator_X, strict=False):
            buffer_BfX[stale_idx] = example_X
            available_indices.add(stale_idx)
            stale_indices.remove(stale_idx)

        # If the buffer wasn't refilled above, the iterator is exhausted, so we yield the remaining activations and break
        if len(available_indices) <= shuffle_buffer_size // 2:
            while len(available_indices) >= yield_batch_size:
                yield sample_BX()
            break

        # yield batches until buffer is half empty
        while len(available_indices) > shuffle_buffer_size // 2:
            yield sample_BX()


T = TypeVar("T")


def _batch_shuffle_iterator(
    iterator: Iterator[T],
    shuffle_buffer_size: int,
    yield_batch_size: int,
    seed: int = 42,
) -> Iterator[list[T]]:
    rng = random.Random(seed)
    refill_threshold = yield_batch_size // 2

    if shuffle_buffer_size < refill_threshold:
        raise ValueError(f"{shuffle_buffer_size=} must be greater than {refill_threshold=}")

    buffer_BfX: list[T] = []

    def refill():
        for item in iterator:
            buffer_BfX.append(item)
            if len(buffer_BfX) >= shuffle_buffer_size:
                break
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


def batch_shuffle_tensor_iterator_BX(
    tensor_iterator_X: Iterator[torch.Tensor],  # X here means "arbitrary shape"
    yield_batch_size_B: int,
    shuffle_buffer_size: int,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    batches_iter = _batch_shuffle_iterator(tensor_iterator_X, yield_batch_size_B, shuffle_buffer_size, seed)
    return (torch.stack(batch) for batch in batches_iter)


if __name__ == "__main__":
    # benchmark
    import time
    from typing import Any

    import torch

    shuffle_buffer_size = 10_000
    yield_batch_size = 10_000

    def new_func(iter_method: Any):
        start = time.time()
        for _ in iter_method(
            (torch.randn(100, 10000) for _ in range(1000)),
            shuffle_buffer_size=shuffle_buffer_size,
            yield_batch_size=yield_batch_size,
        ):
            pass
        end = time.time()
        print(f"Time taken: {end - start} seconds")

    new_func(batch_shuffle_tensor_iterator_BX)
    new_func(batch_shuffle_tensor_iterator_BX_list)
    new_func(batch_shuffle_tensor_iterator_BX_FINAL)