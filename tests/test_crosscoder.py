import torch as t

from model_diffing.models.crosscoder import build_l1_crosscoder


def test_return_shapes():
    n_models = 2
    batch_size = 4
    n_layers = 6
    d_model = 16
    cc_hidden_dim = 256

    crosscoder = build_l1_crosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=cc_hidden_dim,
    )

    activations_BMLD = t.randn(batch_size, n_models, n_layers, d_model)
    y_BLD = crosscoder.forward(activations_BMLD)
    assert y_BLD.shape == activations_BMLD.shape
    train_res = crosscoder.forward_train(activations_BMLD)
    assert train_res.reconstructed_acts_BMLD.shape == activations_BMLD.shape
    assert train_res.hidden_BH.shape == (batch_size, cc_hidden_dim)
