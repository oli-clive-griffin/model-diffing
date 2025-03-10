from einops import einsum
import torch as t

from model_diffing.models import InitStrategy
from model_diffing.models.activations.activation_function import ActivationFunction
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.diffing_crosscoder import N_MODELS, DiffingCrosscoder
from model_diffing.scripts.feb_diff_l1.run import ModelDiffingAnthropicTransposeInit
from model_diffing.utils import l2_norm

# test_weights_rescaling_retains_output
# test_weights_rescaling_max_norm


def assert_close(a: t.Tensor, b: t.Tensor, rtol: float = 1e-05, atol: float = 1e-08):
    assert t.allclose(a, b, rtol=rtol, atol=atol), (
        f"max diff: abs: {t.max(t.abs(a - b)).item():.2e}, rel: {t.max(t.abs(a - b) / t.abs(b)).item():.2e}"
    )


def test_return_shapes():
    batch_size = 4
    d_model = 16
    cc_hidden_dim = 256
    n_explicitly_shared_latents = cc_hidden_dim // 16
    dec_init_norm = 1

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        n_explicitly_shared_latents=n_explicitly_shared_latents,
        hidden_activation=ReLUActivation(),
        init_strategy=ModelDiffingAnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    activations_BMD = t.randn(batch_size, N_MODELS, d_model)
    y_BPD = crosscoder.forward(activations_BMD)
    assert y_BPD.shape == activations_BMD.shape
    train_res = crosscoder.forward_train(activations_BMD)
    assert train_res.recon_acts_BMD.shape == activations_BMD.shape


def test_weights_folding_keeps_hidden_representations_consistent():
    batch_size = 1
    d_model = 5
    cc_hidden_dim = 16
    n_explicitly_shared_latents = 1
    dec_init_norm = 1

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        n_explicitly_shared_latents=n_explicitly_shared_latents,
        hidden_activation=ReLUActivation(),
        init_strategy=ModelDiffingAnthropicTransposeInit(dec_init_norm=dec_init_norm),
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
    cc_hidden_dim = 16
    n_explicitly_shared_latents = 4
    dec_init_norm = 0.1

    cc = DiffingCrosscoder(
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        n_explicitly_shared_latents=n_explicitly_shared_latents,
        hidden_activation=ReLUActivation(),
        init_strategy=ModelDiffingAnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    W_dec_HMD = cc.theoretical_decoder_W_dec_HMD()
    assert W_dec_HMD.shape == (cc_hidden_dim, N_MODELS, d_model)

    W_dec_m0_HsD = W_dec_HMD[:n_explicitly_shared_latents, 0]
    W_dec_m1_HsD = W_dec_HMD[:n_explicitly_shared_latents, 1]
    assert W_dec_m0_HsD.shape == (n_explicitly_shared_latents, d_model)
    assert W_dec_m1_HsD.shape == (n_explicitly_shared_latents, d_model)
    assert (W_dec_m0_HsD == W_dec_m1_HsD).all()

    W_dec_HiMD = W_dec_HMD[n_explicitly_shared_latents:]
    assert W_dec_HiMD.shape == (cc_hidden_dim - n_explicitly_shared_latents, N_MODELS, d_model)

    # check that the shared latents are the first n_explicitly_shared_latents

def test_weights_folding_scales_output_correctly():
    batch_size = 2
    d_model = 6
    cc_hidden_dim = 16
    n_explicitly_shared_latents = 4
    dec_init_norm = 0.1

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        n_explicitly_shared_latents=n_explicitly_shared_latents,
        hidden_activation=ReLUActivation(),
        init_strategy=ModelDiffingAnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    scaling_factors_M = (t.rand(N_MODELS) / 10) + 0.8  # uniformly between 0.8 and 0.9
    scaling_factors_M1 = scaling_factors_M.unsqueeze(-1)

    unscaled_input_BMD = t.randn(batch_size, N_MODELS, d_model)
    scaled_input_BMD = unscaled_input_BMD *  scaling_factors_M1

    scaled_output_BMD = crosscoder.forward_train(scaled_input_BMD).recon_acts_BMD

    crosscoder.fold_activation_scaling_into_weights_(scaling_factors_M)
    unscaled_output_folded_BMD = crosscoder.forward_train(unscaled_input_BMD).recon_acts_BMD
    scaled_output_folded_BMD = unscaled_output_folded_BMD *  scaling_factors_M1

    # with folded weights, the output should be scaled by the scaling factors
    print(l2_norm(scaled_output_BMD - scaled_output_folded_BMD))
    assert_close(scaled_output_BMD, scaled_output_folded_BMD)

class RandomInit(InitStrategy[DiffingCrosscoder[ActivationFunction]]):
    @t.no_grad()
    def init_weights(self, cc: DiffingCrosscoder[ActivationFunction]) -> None:
        cc.W_enc_MDH.random_()
        cc.b_enc_H.random_()

        cc._W_dec_indep_HiMD.random_()
        cc._W_dec_shared_m0_HsD.random_()

        assert (cc._W_dec_shared_m0_HsD == cc._W_dec_shared_m1_HsD).all(), "sanity"

        cc.b_dec_MD.random_()


def test_weights_rescaling_retains_output():
    batch_size = 1
    d_model = 4
    cc_hidden_dim = 8
    n_explicitly_shared_latents = 1

    crosscoder = DiffingCrosscoder(
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        n_explicitly_shared_latents=n_explicitly_shared_latents,
        hidden_activation=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    activations_BMD = t.randn(batch_size, N_MODELS, d_model)

    output_BMD = crosscoder.forward_train(activations_BMD)
    output_rescaled_BMD = crosscoder.with_decoder_unit_norm().forward_train(activations_BMD)

    assert_close(output_BMD.recon_acts_BMD, output_rescaled_BMD.recon_acts_BMD)


def test_weights_rescaling_max_norm():
    d_model = 4
    cc_hidden_dim = 8
    n_explicitly_shared_latents = 1

    cc = DiffingCrosscoder(
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        n_explicitly_shared_latents=n_explicitly_shared_latents,
        hidden_activation=ReLUActivation(),
        init_strategy=RandomInit(),
    ).with_decoder_unit_norm()

    cc_dec_norms_HM = l2_norm(cc._W_dec_indep_HiMD, dim=-1)  # dec norms for each output vector space

    assert (
        t.isclose(cc_dec_norms_HM, t.tensor(1.0))  # for each cc hidden dim,
        .sum(dim=1)  # only 1 output vector space should have norm 1
        .long()
        == t.ones(cc_hidden_dim - n_explicitly_shared_latents).long()
    ).all()
