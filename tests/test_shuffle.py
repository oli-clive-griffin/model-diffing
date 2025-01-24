import torch

from model_diffing.dataloader.shuffle import batch_shuffle_tensor_iterator_BX


def test_shuffle():
    num_items = 100
    item_shape = (2, 3, 4)

    items = [torch.randn(item_shape) for _ in range(num_items)]

    shuffle_buffer_size = 20
    yield_batch_size = 5

    shuffled_iterator = batch_shuffle_tensor_iterator_BX(
        iter(items),
        shuffle_buffer_size,
        yield_batch_size,
    )

    items_batches = list(shuffled_iterator)

    assert items_batches[0][0].shape == item_shape
    assert len(items_batches) == num_items // yield_batch_size


def test_shuffle_uniq():
    num_items = 5
    items_N = torch.arange(num_items)

    shuffled_iterator = batch_shuffle_tensor_iterator_BX(
        iter(items_N),
        shuffle_buffer_size=2,
        yield_batch_size=1,
    )

    items_batches_N = torch.concat(list(shuffled_iterator), dim=0)

    assert items_batches_N.shape == (num_items,), f"{items_batches_N.shape=} != {(num_items,)}"
    assert torch.allclose(torch.tensor(sorted(items_batches_N.tolist())), items_N)
