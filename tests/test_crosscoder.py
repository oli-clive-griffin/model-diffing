import torch as t

from model_diffing.models.crosscoder import AcausalCrosscoder


def test():
    n_models = 2
    batch_size = 4
    n_layers = 6
    d_model = 16
    hidden_dim = 256

    crosscoder = AcausalCrosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=hidden_dim,
        dec_init_norm=0.1,
    )
    activations_NMLD = t.randn(batch_size, n_models, n_layers, d_model)  # , dtype=DTYPE)
    y_NLD = crosscoder.forward(activations_NMLD)
    assert y_NLD.shape == activations_NMLD.shape
    y_NLD, loss = crosscoder.forward_train(activations_NMLD)
    assert y_NLD.shape == activations_NMLD.shape
    assert loss.reconstruction_loss.shape == ()
    assert loss.sparsity_loss.shape == ()
