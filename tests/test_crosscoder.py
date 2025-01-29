import torch as t

from model_diffing.models.crosscoder import BatchTopkActivation, build_relu_crosscoder


def test_return_shapes():
    n_models = 2
    batch_size = 4
    n_layers = 6
    d_model = 16
    cc_hidden_dim = 256
    dec_init_norm = 1

    crosscoder = build_relu_crosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
    )

    activations_BMLD = t.randn(batch_size, n_models, n_layers, d_model)
    y_BLD = crosscoder.forward(activations_BMLD)
    assert y_BLD.shape == activations_BMLD.shape
    train_res = crosscoder.forward_train(activations_BMLD)
    assert train_res.reconstructed_acts_BMLD.shape == activations_BMLD.shape
    assert train_res.hidden_BH.shape == (batch_size, cc_hidden_dim)

def test_batch_topk_activation():
    batch_topk_activation = BatchTopkActivation(k_per_example=2)
    hidden_preactivation_BH = t.tensor([[1, 2, 3, 4, 10], [1, 2, 11, 12, 13]])
    hidden_BH = batch_topk_activation.forward(hidden_preactivation_BH)
    assert hidden_BH.shape == hidden_preactivation_BH.shape
    assert t.all(hidden_BH == t.tensor([[0, 0, 0, 0, 10], [0, 0, 11, 12, 13]]))
