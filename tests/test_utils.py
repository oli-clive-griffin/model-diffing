import torch
from einops import reduce

from model_diffing.utils import l1_norm, l2_norm, multi_reduce


def test_multi_reduce():
    x_ABC = torch.randn(3, 4, 5)

    out_1 = multi_reduce(x_ABC, "a b c", [("c", torch.mean), ("b", l2_norm), ("a", l1_norm)])

    out_2_AB = reduce(x_ABC, "a b c -> a b", torch.mean)
    out_2_A = reduce(out_2_AB, "a b -> a", l2_norm)
    out_2 = reduce(out_2_A, "a ->", l1_norm)

    assert (out_1 == out_2).all()
