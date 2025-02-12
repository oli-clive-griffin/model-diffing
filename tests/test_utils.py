import torch
from einops import reduce

from model_diffing.utils import calculate_explained_variance_X, l1_norm, l2_norm, multi_reduce


def test_multi_reduce():
    x_ABC = torch.randn(3, 4, 5)

    out_1 = multi_reduce(x_ABC, "a b c", ("c", torch.mean), ("b", l2_norm), ("a", l1_norm))

    out_2_AB = reduce(x_ABC, "a b c -> a b", torch.mean)
    out_2_A = reduce(out_2_AB, "a b -> a", l2_norm)
    out_2 = reduce(out_2_A, "a ->", l1_norm)

    assert (out_1 == out_2).all()


def test_explained_variance():
    B = 1
    M = 10
    P = 10
    D = 20
    y_BMPD = torch.randn(B, M, P, D) * 0.1
    y_hat_BMPD = torch.zeros(B, M, P, D)
    ev_MP = calculate_explained_variance_X(y_BMPD, y_hat_BMPD)
    assert ev_MP.shape == (M, P)
    assert ev_MP.mean().item() < 1e-4
    # evs = ev.flatten()
    # plotly.express.histogram(evs, nbins=10).show()
