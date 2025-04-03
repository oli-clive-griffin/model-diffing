import torch

from crosscode.models import (
    AnthropicTransposeInit,
    BatchTopkActivation,
    ModelHookpointAcausalCrosscoder,
    ReLUActivation,
)
from crosscode.models.cross_layer_transcoder import CrossLayerTranscoder
from crosscode.models.initialization.anthropic_transpose import AnthropicTransposeInitCrossLayerTC


def test_return_shapes():
    n_models = 2
    batch_size = 4
    n_hookpoints = 6
    d_model = 16
    n_latents = 256
    dec_init_norm = 1

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=n_models,
        n_hookpoints=n_hookpoints,
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    activations_BMPD = torch.randn(batch_size, n_models, n_hookpoints, d_model)
    assert crosscoder.forward(activations_BMPD).shape == activations_BMPD.shape
    train_res = crosscoder.forward_train(activations_BMPD)
    assert train_res.recon_acts_BMPD.shape == activations_BMPD.shape
    assert train_res.latents_BL.shape == (batch_size, n_latents)


def test_batch_topk_activation():
    batch_topk_activation = BatchTopkActivation(k_per_example=2)
    hidden_preact_BL = torch.tensor([[1, 2, 3, 4, 10], [1, 2, 11, 12, 13]])
    hidden_BL = batch_topk_activation.forward(hidden_preact_BL)
    assert hidden_BL.shape == hidden_preact_BL.shape
    assert torch.all(hidden_BL == torch.tensor([[0, 0, 0, 0, 10], [0, 0, 11, 12, 13]]))


def test_weights_folding_keeps_hidden_representations_consistent():
    batch_size = 1
    n_models = 3
    n_hookpoints = 4
    d_model = 5
    n_latents = 16
    dec_init_norm = 1

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=n_models,
        n_hookpoints=n_hookpoints,
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
        use_encoder_bias=True,
        use_decoder_bias=True,
    )

    scaling_factors_MP = torch.randn(n_models, n_hookpoints)

    unscaled_input_BMPD = torch.randn(batch_size, n_models, n_hookpoints, d_model)
    scaled_input_BMPD = unscaled_input_BMPD * scaling_factors_MP[..., None]

    output_without_folding = crosscoder.forward_train(scaled_input_BMPD)

    output_with_folding = crosscoder.with_folded_scaling_factors(scaling_factors_MP).forward_train(unscaled_input_BMPD)

    # all hidden representations should be the same
    assert torch.allclose(output_without_folding.latents_BL, output_with_folding.latents_BL), (
        f"max diff: {torch.max(torch.abs(output_without_folding.latents_BL - output_with_folding.latents_BL))}"
    )

    output_after_unfolding = crosscoder.forward_train(scaled_input_BMPD)

    assert torch.allclose(output_without_folding.latents_BL, output_after_unfolding.latents_BL), (
        f"max diff: {torch.max(torch.abs(output_without_folding.latents_BL - output_after_unfolding.latents_BL))}"
    )


def test_weights_folding_scales_output_correctly():
    batch_size = 2

    n_models = 2
    n_hookpoints = 1

    d_model = 6
    n_latents = 6
    dec_init_norm = 0.1

    crosscoder = ModelHookpointAcausalCrosscoder(
        n_models=n_models,
        n_hookpoints=n_hookpoints,
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    # scaling_factors_MP = torch.randn(n_models, n_hookpoints)
    scaling_factors_MP = (torch.rand(n_models, n_hookpoints) / 10) + 0.8  # 0.8 to 0.9
    scaling_factors_MP1 = scaling_factors_MP.unsqueeze(-1)

    unscaled_input_BMPD = torch.randn(batch_size, n_models, n_hookpoints, d_model)
    scaled_input_BMPD = unscaled_input_BMPD * scaling_factors_MP1

    scaled_output_BMPD = crosscoder.forward_train(scaled_input_BMPD).recon_acts_BMPD

    crosscoder.fold_activation_scaling_into_weights_(scaling_factors_MP)
    unscaled_output_folded_BMPD = crosscoder.forward_train(unscaled_input_BMPD).recon_acts_BMPD

    # with folded weights, the output should be scaled by the scaling factors
    torch.testing.assert_close(scaled_output_BMPD, unscaled_output_folded_BMPD * scaling_factors_MP1)


def test_weights_folding_scales_output_correctly_tc():
    batch_size = 2
    d_model = 4
    n_layers_out = 3
    n_latents = 6
    dec_init_norm = 0.1

    crosscoder = CrossLayerTranscoder(
        n_layers_out=n_layers_out,
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=AnthropicTransposeInitCrossLayerTC(dec_init_norm=dec_init_norm),
    )

    # scaling_factors_MP = torch.randn(n_models, n_hookpoints)
    scaling_factors_P = (torch.rand(1 + n_layers_out) / 10) + 0.8  # 0.8 to 0.9
    scaling_factor_in = scaling_factors_P[0]
    scaling_factors_out_Po = scaling_factors_P[1:]

    # input shape: (batch_size, d_model)

    unscaled_input_BD = torch.randn(batch_size, d_model)
    scaled_input_BD = unscaled_input_BD * scaling_factor_in

    scaled_output = crosscoder.forward_train(scaled_input_BD)
    unscaled_output_folded = crosscoder.with_folded_scaling_factors(scaling_factors_P).forward_train(unscaled_input_BD)

    # with folded weights, the output should be scaled by the scaling factors
    torch.testing.assert_close(
        scaled_output.output_BPD,
        unscaled_output_folded.output_BPD * scaling_factors_out_Po[..., None],
    )

    # and the latents should be the same
    torch.testing.assert_close(scaled_output.latents_BL, unscaled_output_folded.latents_BL)
