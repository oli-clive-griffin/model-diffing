from typing import Any

import torch

from crosscoding.models import (
    AnthropicTransposeInit,
    BatchTopkActivation,
    InitStrategy,
    ModelHookpointAcausalCrosscoder,
    ReLUActivation,
)
from crosscoding.utils import l2_norm


def test_return_shapes():
    n_models = 2
    batch_size = 4
    n_hookpoints = 6
    d_model = 16
    n_latents = 256
    dec_init_norm = 1

    crosscoder = ModelHookpointAcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
        use_encoder_bias=True,
        use_decoder_bias=True,
    )

    activations_BMPD = torch.randn(batch_size, n_models, n_hookpoints, d_model)
    y_BPD = crosscoder.forward(activations_BMPD)
    assert y_BPD.shape == activations_BMPD.shape
    train_res = crosscoder.forward_train(activations_BMPD)
    assert train_res.recon_acts_BXD.shape == activations_BMPD.shape
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
        crosscoding_dims=(n_models, n_hookpoints),
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

    output_with_folding = crosscoder.with_folded_scaling_factors(scaling_factors_MP, scaling_factors_MP).forward_train(
        unscaled_input_BMPD
    )

    # all hidden representations should be the same
    assert torch.allclose(output_without_folding.latents_BL, output_with_folding.latents_BL), (
        f"max diff: {torch.max(torch.abs(output_without_folding.latents_BL - output_with_folding.latents_BL))}"
    )

    output_after_unfolding = crosscoder.forward_train(scaled_input_BMPD)

    assert torch.allclose(output_without_folding.latents_BL, output_after_unfolding.latents_BL), (
        f"max diff: {torch.max(torch.abs(output_without_folding.latents_BL - output_after_unfolding.latents_BL))}"
    )


def assert_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08):
    assert torch.allclose(a, b, rtol=rtol, atol=atol), (
        f"max diff: abs: {torch.max(torch.abs(a - b)).item():.2e}, rel: {torch.max(torch.abs(a - b) / torch.abs(b)).item():.2e}"
    )


def test_weights_folding_scales_output_correctly():
    batch_size = 2

    # TODO UNDO ME
    n_models = 2
    n_hookpoints = 1

    d_model = 6
    n_latents = 6
    dec_init_norm = 0.1

    crosscoder = ModelHookpointAcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
        use_encoder_bias=True,
        use_decoder_bias=True,
    )

    # scaling_factors_MP = torch.randn(n_models, n_hookpoints)
    scaling_factors_MP = (torch.rand(n_models, n_hookpoints) / 10) + 0.8  # 0.8 to 0.9
    scaling_factors_MP1 = scaling_factors_MP.unsqueeze(-1)

    unscaled_input_BMPD = torch.randn(batch_size, n_models, n_hookpoints, d_model)
    scaled_input_BMPD = unscaled_input_BMPD * scaling_factors_MP1

    scaled_output_BMPD = crosscoder.forward_train(scaled_input_BMPD).recon_acts_BXD

    crosscoder.fold_activation_scaling_into_weights_(scaling_factors_MP, scaling_factors_MP)
    unscaled_output_folded_BMPD = crosscoder.forward_train(unscaled_input_BMPD).recon_acts_BXD

    # with folded weights, the output should be scaled by the scaling factors
    assert_close(scaled_output_BMPD, unscaled_output_folded_BMPD * scaling_factors_MP1)


class RandomInit(InitStrategy[ModelHookpointAcausalCrosscoder[Any]]):
    @torch.no_grad()
    def init_weights(self, cc: ModelHookpointAcausalCrosscoder[Any]) -> None:
        cc.W_enc_XDL.normal_()
        if cc.b_enc_L is not None:
            cc.b_enc_L.zero_()

        cc.W_dec_LXD.normal_()
        if cc.b_dec_XD is not None:
            cc.b_dec_XD.zero_()


def test_weights_rescaling_retains_output():
    batch_size = 1
    n_models = 2
    n_hookpoints = 3
    d_model = 4
    n_latents = 8

    crosscoder = ModelHookpointAcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
        use_encoder_bias=True,
        use_decoder_bias=True,
    )

    activations_BMPD = torch.randn(batch_size, n_models, n_hookpoints, d_model)

    train_res = crosscoder.forward_train(activations_BMPD)
    output_rescaled_BMPD = crosscoder.with_decoder_unit_norm().forward_train(activations_BMPD)

    assert torch.allclose(train_res.recon_acts_BXD, output_rescaled_BMPD.recon_acts_BXD), (
        f"max diff: {torch.max(torch.abs(train_res.recon_acts_BXD - output_rescaled_BMPD.recon_acts_BXD))}"
    )


def test_weights_rescaling_max_norm():
    n_models = 2
    n_hookpoints = 3
    d_model = 4
    n_latents = 8

    cc = ModelHookpointAcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        n_latents=n_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
        use_encoder_bias=True,
        use_decoder_bias=True,
    ).with_decoder_unit_norm()

    cc_dec_norms_LMP = l2_norm(cc.W_dec_LXD, dim=-1)  # dec norms for each output vector space

    assert torch.allclose(
        torch.isclose(cc_dec_norms_LMP, torch.tensor(1.0))  # for each cc hidden dim,
        .sum(dim=(1, 2))  # only 1 output vector space should have norm 1
        .long(),
        torch.ones(n_latents).long(),
    )
