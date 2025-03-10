import tempfile
from pathlib import Path

import torch as t

from model_diffing.models import InitStrategy
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.diffing_crosscoder import N_MODELS, DiffingCrosscoder
from model_diffing.utils import l2_norm


class RandomInit(InitStrategy[DiffingCrosscoder[ActivationFunction]]):
    @t.no_grad()
    def init_weights(self, cc: DiffingCrosscoder[ActivationFunction]) -> None:
        cc.W_enc_MDH.normal_()
        cc.b_enc_H.zero_()

        cc._W_dec_indep_HiMD.normal_()
        cc._W_dec_shared_m0_HsD.normal_()
        assert (cc._W_dec_shared_m0_HsD == cc._W_dec_shared_m1_HsD).all(), "sanity check for tied weights"

        cc.b_dec_MD.zero_()


# looser tolerance for default float32
def assert_close(a: t.Tensor, b: t.Tensor, rtol: float = 1e-04, atol: float = 1e-04):
    assert t.allclose(a, b, rtol=rtol, atol=atol), (
        f"max diff: abs: {t.max(t.abs(a - b)).item():.2e}, rel: {t.max(t.abs(a - b) / t.abs(b)).item():.2e}"
    )


def test_return_shapes():
    batch_size = 4
    d_model = 16
    n_latents = 32
    n_explicitly_shared_latents = 2

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=n_latents,
        n_shared_latents=n_explicitly_shared_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    activations_BMD = t.randn(batch_size, N_MODELS, d_model)
    y_BPD = crosscoder.forward(activations_BMD)
    assert y_BPD.shape == activations_BMD.shape
    train_res = crosscoder.forward_train(activations_BMD)
    assert train_res.recon_acts_BMD.shape == activations_BMD.shape


def test_weights_folding_keeps_hidden_representations_consistent():
    batch_size = 1
    d_model = 5
    n_latents = 32
    n_explicitly_shared_latents = 2

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=n_latents,
        n_shared_latents=n_explicitly_shared_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    scaling_factors_M = t.randn(N_MODELS)

    unscaled_input_BMD = t.randn(batch_size, N_MODELS, d_model)
    scaled_input_BMD = unscaled_input_BMD * scaling_factors_M[..., None]

    output_without_folding = crosscoder.forward_train(scaled_input_BMD)

    with crosscoder.temporarily_fold_activation_scaling(scaling_factors_M):
        output_with_folding = crosscoder.forward_train(unscaled_input_BMD)

    # all hidden representations should be the same
    assert_close(output_without_folding.hidden_indep_BHi, output_with_folding.hidden_indep_BHi)
    assert_close(output_without_folding.hidden_shared_BHs, output_with_folding.hidden_shared_BHs)

    output_after_unfolding = crosscoder.forward_train(scaled_input_BMD)

    assert_close(output_without_folding.hidden_indep_BHi, output_after_unfolding.hidden_indep_BHi)
    assert_close(output_without_folding.hidden_shared_BHs, output_after_unfolding.hidden_shared_BHs)


def test_theoretical_decoder_W_dec_HMD():
    d_model = 4

    n_latents = 32
    n_explicitly_shared_latents = 2

    cc = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=n_latents,
        n_shared_latents=n_explicitly_shared_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    W_dec_HMD = cc.theoretical_decoder_W_dec_HMD()
    assert W_dec_HMD.shape == (n_latents, N_MODELS, d_model)

    W_dec_m0_HsD = W_dec_HMD[:n_explicitly_shared_latents, 0]
    W_dec_m1_HsD = W_dec_HMD[:n_explicitly_shared_latents, 1]
    assert W_dec_m0_HsD.shape == (n_explicitly_shared_latents, d_model)
    assert W_dec_m1_HsD.shape == (n_explicitly_shared_latents, d_model)
    assert (W_dec_m0_HsD == W_dec_m1_HsD).all()

    W_dec_HiMD = W_dec_HMD[n_explicitly_shared_latents:]
    assert W_dec_HiMD.shape == (n_latents - n_explicitly_shared_latents, N_MODELS, d_model)

    # check that the shared latents are the first n_explicitly_shared_latents


def test_weights_folding_scales_output_correctly():
    batch_size = 2
    d_model = 6
    n_latents = 32
    n_explicitly_shared_latents = 2

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=n_latents,
        n_shared_latents=n_explicitly_shared_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    scaling_factors_M = (t.rand(N_MODELS) / 10) + 0.8  # uniformly between 0.8 and 0.9
    scaling_factors_M1 = scaling_factors_M.unsqueeze(-1)

    unscaled_input_BMD = t.randn(batch_size, N_MODELS, d_model)
    scaled_input_BMD = unscaled_input_BMD * scaling_factors_M1

    scaled_output_BMD = crosscoder.forward_train(scaled_input_BMD).recon_acts_BMD

    crosscoder.fold_activation_scaling_into_weights_(scaling_factors_M)
    unscaled_output_folded_BMD = crosscoder.forward_train(unscaled_input_BMD).recon_acts_BMD
    scaled_output_folded_BMD = unscaled_output_folded_BMD * scaling_factors_M1

    # with folded weights, the output should be scaled by the scaling factors
    print(l2_norm(scaled_output_BMD - scaled_output_folded_BMD))
    assert_close(scaled_output_BMD, scaled_output_folded_BMD)


def test_weights_rescaling_retains_output():
    batch_size = 1
    d_model = 8
    n_latents = 32
    n_explicitly_shared_latents = 2

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=n_latents,
        n_shared_latents=n_explicitly_shared_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    activations_BMD = t.randn(batch_size, N_MODELS, d_model)

    output_BMD = crosscoder.forward_train(activations_BMD)
    output_rescaled_BMD = crosscoder.with_decoder_unit_norm().forward_train(activations_BMD)

    assert_close(output_BMD.recon_acts_BMD, output_rescaled_BMD.recon_acts_BMD)


def test_decoder_unit_norm():
    """
    `DiffingCrosscoder.with_decoder_unit_norm()` should ensure that the maximum decoder norm for a given latent is 1
    """

    d_model = 4
    n_latents = 32
    n_explicitly_shared_latents = 2

    cc = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=n_latents,
        n_shared_latents=n_explicitly_shared_latents,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    ).with_decoder_unit_norm()

    cc_dec_norms_HM = l2_norm(cc._W_dec_indep_HiMD, dim=-1)  # dec norms for each output vector space

    assert (
        t.isclose(cc_dec_norms_HM, t.tensor(1.0))  # for each cc hidden dim,
        .sum(dim=1)  # only 1 output vector space should have norm 1
        .long()
        == t.ones(n_latents - n_explicitly_shared_latents).long()
    ).all()


def asdf():
    d_model = 4

    cc = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=32,
        n_shared_latents=2,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    )
    scaling_factors_M = (t.rand(N_MODELS) / 10) + 0.8
    raw_activations_BMD = t.randn(1, N_MODELS, d_model)

    output_before = cc.forward_train(raw_activations_BMD * scaling_factors_M[..., None])
    with cc.temporarily_fold_activation_scaling(scaling_factors_M):
        during_recon_acts_BMD = cc.forward(raw_activations_BMD)
    output_after = cc.forward_train(raw_activations_BMD * scaling_factors_M[..., None])

    assert_close(output_before.recon_acts_BMD, output_after.recon_acts_BMD)
    assert_close(output_before.recon_acts_BMD, during_recon_acts_BMD * scaling_factors_M[..., None])


def test_e2e():
    d_model = 4

    # GIVEN:
    # - a diffing crosscoder
    cc = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=32,
        n_shared_latents=2,
        activation_fn=ReLUActivation(),
        init_strategy=RandomInit(),
    )
    # - some non-unit scaling factors (distributed uniformly between 0.8 and 0.9)
    scaling_factors_M = (t.rand(N_MODELS) / 10) + 0.8
    # - some raw activations
    raw_activations_BMD = t.randn(1, N_MODELS, d_model)
    # - and a version of the same model which has been saved and loaded with the scaling factors folded into the weights
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model"
        with cc.temporarily_fold_activation_scaling(scaling_factors_M):
            during_acts_BMD = cc.forward(raw_activations_BMD)
            cc.save(model_path)
            cc_loaded_folded = DiffingCrosscoder.load(model_path)
            pass
    scaled_activations_BMD = raw_activations_BMD * scaling_factors_M[..., None]

    # WHEN:
    # we run the original model on scaled activations, and the loaded model on raw activations
    train_output = cc.forward_train(scaled_activations_BMD)
    loaded_output = cc_loaded_folded.forward_train(raw_activations_BMD)

    # THEN:
    # the output from the loaded model should be equal to the output from
    # the original model, but scaled by the scaling factors.
    assert_close(train_output.recon_acts_BMD, during_acts_BMD * scaling_factors_M[..., None])
    assert_close(train_output.recon_acts_BMD, loaded_output.recon_acts_BMD * scaling_factors_M[..., None])
    

    # While the hidden representations should be the same
    assert_close(train_output.hidden_indep_BHi, loaded_output.hidden_indep_BHi)
    assert_close(train_output.hidden_shared_BHs, loaded_output.hidden_shared_BHs)

    # AND WHEN
    # we set the decoder norms to 1
    # cc_loaded_folded_unit_norm = cc_loaded_folded.with_decoder_unit_norm()
    # loaded_output_unit_norm = cc_loaded_folded_unit_norm.forward_train(raw_activations_BMD)

    cc_loaded_folded.make_decoder_max_unit_norm_()
    loaded_output_unit_norm = cc_loaded_folded.forward_train(raw_activations_BMD)

    # THEN:
    # the output should be the same as before
    assert_close(loaded_output.recon_acts_BMD, loaded_output_unit_norm.recon_acts_BMD)

    # While the hidden representations should be the same
    assert_close(loaded_output.hidden_indep_BHi, loaded_output_unit_norm.hidden_indep_BHi)
    assert_close(loaded_output.hidden_shared_BHs, loaded_output_unit_norm.hidden_shared_BHs)


def asdf():
    d_model = 4

    scaling_factors_M = t.tensor([0.5, 0.2])
    cc_folded = DiffingCrosscoder(
        d_model=d_model,
        n_latents_total=32,
        n_shared_latents=4,
        init_strategy=RandomInit(),
        activation_fn=ReLUActivation(),
    ).with_activation_scaling(scaling_factors_M)
    # - some non-unit scaling factors (distributed uniformly between 0.8 and 0.9)
    # - and a version of the same model which has been saved and loaded with the scaling factors folded into the weights

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model"
        cc_folded.save(model_path)
        cc_folded_loaded = DiffingCrosscoder.load(model_path)

    assert t.allclose(cc_folded.W_enc_MDH, cc_folded_loaded.W_enc_MDH)
    assert t.allclose(cc_folded.b_enc_H, cc_folded_loaded.b_enc_H)

    assert t.allclose(cc_folded._W_dec_indep_HiMD, cc_folded_loaded._W_dec_indep_HiMD)
    assert t.allclose(cc_folded._W_dec_shared_m1_HsD, cc_folded_loaded._W_dec_shared_m1_HsD)
    assert t.allclose(cc_folded._W_dec_shared_m0_HsD, cc_folded_loaded._W_dec_shared_m0_HsD)

    
    # cc._W_dec_shared_m1_HsD[:3, :3] / cc_folded_loaded._W_dec_shared_m1_HsD[:3, :3]

    assert t.allclose(cc_folded.b_dec_MD, cc_folded_loaded.b_dec_MD)

asdf()