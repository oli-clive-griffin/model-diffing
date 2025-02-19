import torch

from model_diffing.scripts.train_jan_update_crosscoder.run import compute_b_enc_H


def test_compute_b_enc_H():
    hidden_dim = 2

    # use an identity matrix for the encoder weights so that the pre-bias is just the feature values
    W_enc_XDH = torch.eye(hidden_dim).float()

    initial_approx_firing_pct = 0.25

    batch_BD = torch.tensor(
        [
            [0, 4],
            [1, 5],
            [2, 6],
            # this is where the threshold should be drawn. 25% of the examples should fire, i.e. just the example [3, 7]
            [3, 7],
        ]
    ).float()

    activations_iterator_BD = iter([batch_BD, batch_BD.clone()])

    # use a jumprelu threshold of 1.0 for simplicity
    initial_jumprelu_threshold_H = torch.ones(hidden_dim).float()

    b_enc_H = compute_b_enc_H(
        activations_iterator_BD,
        W_enc_XDH,
        initial_jumprelu_threshold_H,
        initial_approx_firing_pct,
        n_examples_to_sample=8,  # just the size of the dataset (2 batches of 4 examples each)
    )

    assert b_enc_H.shape == (hidden_dim,)

    # the threshold should be the 75th percentile of the pre-bias values, minus the jumprelu threshold
    # - 75th quantile of the pre-bias values is [2.25, 6.25] (see linear interpolation in torch.quantile)
    # - the jumprelu threshold is 1.0, so the threshold is 1 - [2.25, 6.25]
    expected_b_enc_H = torch.tensor([1 - 2.25, 1 - 6.25])
    assert torch.allclose(b_enc_H, expected_b_enc_H), f"b_enc_H: {b_enc_H}, expected_b_enc_H: {expected_b_enc_H}"


test_compute_b_enc_H()


def test_compute_b_enc_H_batches_rounding():
    hidden_dim = 1

    # use an identity matrix for the encoder weights so that the pre-bias is just the feature values
    W_enc_XDH = torch.eye(hidden_dim).float()  # this is just [1.], but wanting to make the identity transform explicit

    initial_approx_firing_pct = 0.25

    batch_1_BD = torch.tensor([[0], [1], [2]]).float()
    batch_2_BD = torch.tensor([[3], [4], [5]]).float()

    activations_iterator_BD = iter([batch_1_BD, batch_2_BD])

    initial_jumprelu_threshold_H = torch.randn(hidden_dim).float()

    b_enc_H = compute_b_enc_H(
        activations_iterator_BD,
        W_enc_XDH,
        initial_jumprelu_threshold_H,
        initial_approx_firing_pct,
        n_examples_to_sample=5,  # should round up to 6 (taking in both batches)
    )

    assert b_enc_H.shape == (hidden_dim,)

    # the threshold should be the 75th percentile of the pre-bias values, minus the jumprelu threshold
    whole_dataset_BD = torch.cat([batch_1_BD, batch_2_BD])
    expected_b_enc_H = initial_jumprelu_threshold_H - torch.quantile(whole_dataset_BD, 1 - initial_approx_firing_pct)

    assert torch.allclose(b_enc_H, expected_b_enc_H), f"b_enc_H: {b_enc_H}, expected_b_enc_H: {expected_b_enc_H}"
