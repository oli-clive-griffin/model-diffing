# %%
from typing import cast

import plotly.express as px  # type: ignore
import torch

from model_diffing.analysis import visualization
from model_diffing.analysis.metrics import get_IQR_outliers_mask
from model_diffing.data.model_layer_dataloader import build_dataloader
from model_diffing.scripts import config_common
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.utils import collect_norms
from model_diffing.utils import get_device

torch.set_grad_enabled(False)

# %%
llm_configs = [
    config_common.LLMConfig(
        name="gpt2",
    ),
]

sequence_iterator_config = config_common.SequenceIteratorConfig(
    classname="HuggingfaceTextDatasetTokenSequenceLoader",
    kwargs={
        "hf_dataset_name": "monology/pile-uncopyrighted",
        "sequence_length": 258,
        "shuffle_buffer_size": 4096,
    },
)

activations_harvester_config = config_common.ActivationsHarvesterConfig(
    llms=llm_configs,
    layer_indices_to_harvest=[0, 3, 7, 9, 11],
    inference_dtype="float32",
    harvesting_batch_size=16,
)

data_config = config_common.DataConfig(
    sequence_iterator=sequence_iterator_config,
    activations_harvester=activations_harvester_config,
    activations_shuffle_buffer_size=1000,
)

cache_dir = "./.cache/norms"

# %%
device = get_device()
llms = build_llms(llm_configs, cache_dir, device, dtype=activations_harvester_config.inference_dtype)

# %%
dataloader = build_dataloader(data_config, 16, cache_dir, device)

# %%
sample_BMLD = next(dataloader.get_shuffled_activations_iterator_BMLD())
print(sample_BMLD.shape, sample_BMLD.device)

# %%
norms_NML = collect_norms(dataloader.get_shuffled_activations_iterator_BMLD(), device=device, n_batches=512)
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
    model_names=cast(list[str], [model.name for model in llms]),
    layer_names=data_config.activations_harvester.layer_indices_to_harvest,
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
    model_names=cast(list[str], [model.name for model in llms]),
    layer_names=data_config.activations_harvester.layer_indices_to_harvest,
    k_iqr=3.0,
    overlay=True,
    log=True,
)

fig = fig_list[0]
fig.update_layout(width=1600)
fig.show()
