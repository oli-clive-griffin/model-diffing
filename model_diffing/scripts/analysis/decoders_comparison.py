# %%
import torch

from model_diffing.analysis import metrics, visualization
from model_diffing.models import crosscoder

torch.set_grad_enabled(False)

# %%

# for example:
# path = ".checkpoints/l1_crosscoder_pythia_160M_layer_3/model_epoch_5000.pt"

path = ""

state_dict = torch.load(path)

cc = crosscoder.AcausalCrosscoder(
    n_models=2,
    n_layers=1,
    d_model=768,
    hidden_dim=6_144,
    dec_init_norm=0.1,
    hidden_activation=crosscoder.ReLUActivation(),
)
cc.load_state_dict(state_dict)

# %%


for layer_idx in range(cc.W_dec_HMLD.shape[2]):
    W_dec_a_HD = cc.W_dec_HMLD[:, 0, layer_idx]
    W_dec_b_HD = cc.W_dec_HMLD[:, 1, layer_idx]

    # %%
    # fig = visualization.plot_relative_norms(W_dec_a_HD, W_dec_b_HD)
    relative_norms = metrics.compute_relative_norms_N(W_dec_a_HD, W_dec_b_HD)
    fig = visualization.relative_norms_hist(relative_norms)
    fig.show()

    cosine_sims_H = metrics.compute_cosine_similarities_N(W_dec_a_HD, W_dec_b_HD)
    fig = visualization.plot_cosine_sim(cosine_sims_H)
    fig.show()

    shared_latent_mask = metrics.get_shared_latent_mask(relative_norms)
    print(shared_latent_mask.shape)

    # # %%
    cosine_sims = metrics.compute_cosine_similarities_N(W_dec_a_HD, W_dec_b_HD)
    print(cosine_sims.shape)

    # # %%
    shared_features_cosine_sims = cosine_sims[shared_latent_mask]
    fig = visualization.plot_cosine_sim(shared_features_cosine_sims)
    fig.show()

# %%
