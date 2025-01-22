# %%
import plotly.express as px
import torch

from model_diffing.analysis import visualization
from model_diffing.analysis.metrics import get_IQR_outliers_mask
from model_diffing.dataloader.data import build_dataloader_BMLD
from model_diffing.scripts import config_common
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.utils import collect_norms
from model_diffing.utils import get_device

torch.set_grad_enabled(False)

# %%
llm_configs = config_common.LLMsConfig(
    models=[
        config_common.LLMConfig(
            name="gpt2",
        ),
    ]
)

sequence_iterator_config = config_common.SequenceTokensIteratorConfig(
    classname="CommonCorpusTokenSequenceIterator", kwargs={"sequence_length": 258}
)

activations_config = config_common.ActivationsIteratorConfig(
    layer_indices_to_harvest=[0, 3, 7, 9, 11], harvest_batch_size=16, sequence_tokens_iterator=sequence_iterator_config
)

data_config = config_common.DataConfig(activations_iterator=activations_config, shuffle_buffer_size=1000, batch_size=16)

cfg = config_common.BaseExperimentConfig(data=data_config, llms=llm_configs, wandb="disabled")

# %%
device = get_device()
llms = build_llms(cfg.llms, cfg.cache_dir, device)

# %%
dataloader_BMLD = build_dataloader_BMLD(cfg.data, llms, cfg.cache_dir)

# %%
sample_BMLD = next(dataloader_BMLD)
print(sample_BMLD.shape, sample_BMLD.device)

# %%
norms_NML = collect_norms(dataloader_BMLD, device=device, n_batches=512)
print(norms_NML.shape, norms_NML.device)

# %%
M, L = 0, 0
norms = norms_NML[:, M, L]

fig = px.histogram(norms.cpu().numpy())
fig.show()

# %%
outliers = get_IQR_outliers_mask(norms, k_iqr=3.0)
norms = norms[~outliers]

fig = px.histogram(norms.cpu().numpy())
fig.show()

# %%
fig_list = visualization.plot_norms_hists(
    norms_NML,
    model_names=[model.name for model in cfg.llms.models],
    layer_names=cfg.data.activations_iterator.layer_indices_to_harvest,
    k_iqr=3.0,
    overlay=False,
    log=False,
)

fig = fig_list[0]
fig.update_layout(width=1600)
fig.show()

# %%
fig_list = visualization.plot_norms_hists(
    norms_NML,
    model_names=[model.name for model in cfg.llms.models],
    layer_names=cfg.data.activations_iterator.layer_indices_to_harvest,
    k_iqr=3.0,
    overlay=True,
    log=True,
)

fig = fig_list[0]
fig.update_layout(width=1600)
fig.show()
