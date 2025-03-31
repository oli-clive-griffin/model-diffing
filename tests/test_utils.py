import numpy as np
import torch

from crosscoding.utils import calculate_vector_norm_fvu_X


def test_fvu_should_be_1():
    B = 10000
    M = 1
    P = 1
    D = 1  # because it's a vector norm based fvu, this makes testing easier

    # When we try to predict random noise by predicting the mean, we expect fvu to be 1
    y_BMPD = torch.rand(B, M, P, D)  # stay in the positive region to avoid awkwardness with norms always being positive
    y_hat_BMPD = torch.ones_like(y_BMPD) * y_BMPD.mean(dim=0, keepdim=True)
    expected_fvu = 1

    fvu = calculate_vector_norm_fvu_X(y_BMPD, y_hat_BMPD).item()
    assert np.isclose(fvu, expected_fvu, atol=1e-2), f"{fvu=}, expected {expected_fvu}"


def test_fvu_should_be_0():
    B = 10000
    M = 1
    P = 1
    D = 1  # because it's a vector norm based fvu, this makes testing easier

    # When we try to predict random noise by predicting the mean, we expect fvu to be 1
    y_BMPD = torch.rand(B, M, P, D)  # stay in the positive region to avoid awkwardness with norms always being positive
    y_hat_BMPD = y_BMPD.clone()
    expected_fvu = 0

    fvu = calculate_vector_norm_fvu_X(y_BMPD, y_hat_BMPD).item()
    assert np.isclose(fvu, expected_fvu, atol=1e-2), f"{fvu=}, expected {expected_fvu}"


def test_fvu_shape():
    B = 10
    M = 2
    P = 3
    D = 4
    y_BMPD = torch.rand(B, M, P, D)
    y_hat_BMPD = torch.ones(B, M, P, D) * 0.5
    fvu_MP = calculate_vector_norm_fvu_X(y_BMPD, y_hat_BMPD)
    assert fvu_MP.shape == (M, P)
