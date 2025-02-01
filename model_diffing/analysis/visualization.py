import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
from plotly.subplots import make_subplots

from model_diffing.analysis import metrics
from model_diffing.log import logger


def create_visualizations(W_dec_HMLD: torch.Tensor, layers: list[int]) -> dict[str, go.Figure] | None:
    _, num_models, num_layers, _ = W_dec_HMLD.shape
    if num_models != 2:
        logger.warning(f"num_models != 2, skipping visualizations. num_models: {num_models}")
        return None

    plots = {}

    for layer_idx in range(num_layers):
        layer_name = layers[layer_idx]  # layer_idx is the index into the list of layers we're collecting
        a_HD = W_dec_HMLD[:, 0, layer_idx]
        b_HD = W_dec_HMLD[:, 1, layer_idx]

        relative_norms_fig = plot_relative_norms(a_HD, b_HD, title=f"Relative Norms. Layer {layer_name}")
        plots[f"relative_decoder_norms_layer_{layer_name}"] = relative_norms_fig

        cosine_sim_fig = plot_cosine_sim(
            metrics.compute_cosine_similarities_N(a_HD, b_HD), title=f"Cosine Similarity. Layer {layer_name}"
        )
        plots[f"cosine_sim_layer_{layer_name}"] = cosine_sim_fig

    return plots


def plot_relative_norms(vectors_a: torch.Tensor, vectors_b: torch.Tensor, title: str | None = None) -> go.Figure:
    """Plot histogram of relative norms (norm_b / (norm_a + norm_b)).

    Args:
        vectors_a: Tensor of vectors from the first set
        vectors_b: Tensor of vectors from the second set
        title: Optional title for the plot

    Returns:
        Plotly figure object
    """
    relative_norms = metrics.compute_relative_norms_N(vectors_a, vectors_b)

    return relative_norms_hist(relative_norms, title=title)


def relative_norms_hist(relative_norms_N: torch.Tensor, title: str | None = None) -> go.Figure:
    fig = px.histogram(
        relative_norms_N.detach().cpu().numpy(),
        nbins=200,
        labels={"value": "Relative norm"},
        title=title,
        range_x=[0, 1],
    )

    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Number of Latents")
    fig.update_xaxes(tickvals=[0, 0.25, 0.5, 0.75, 1.0], ticktext=["0", "0.25", "0.5", "0.75", "1.0"])
    return fig


def plot_cosine_sim(cosine_sims_N: torch.Tensor, title: str | None = None) -> go.Figure:
    """Plot histogram of cosine similarities.

    Args:
        cosine_sims: Tensor of cosine similarity values
        title: Optional title for the plot

    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        cosine_sims_N.detach().cpu().numpy(),
        log_y=True,
        range_x=[-1, 1],
        nbins=100,
        labels={"value": "Cosine similarity"},
        title=title,
    )

    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Number of Latents (log scale)")
    return fig


def _build_df(
    values_NML: torch.Tensor,
    model_names: list[str] | None = None,
    layer_names: list[int] | list[str] | None = None,
    k_iqr: float | None = None,
) -> pd.DataFrame:
    df_list = []

    for model_idx in range(values_NML.shape[1]):
        for layer_idx in range(values_NML.shape[2]):
            values = values_NML[:, model_idx, layer_idx]

            if k_iqr:
                outliers = metrics.get_IQR_outliers_mask(values.unsqueeze(1), k_iqr=k_iqr).squeeze()
                values = values[~outliers]

            df_list.append(
                pd.DataFrame(
                    {
                        "Values": values.cpu().numpy(),
                        "Layer": layer_idx,
                        "Model": model_idx,
                    }
                )
            )

    df = pd.concat(df_list, ignore_index=True)

    # Map model and layer indices to the provided names
    if model_names:
        df["Model"] = df["Model"].replace(dict(enumerate(model_names)))
    if layer_names:
        df["Layer"] = df["Layer"].replace(dict(enumerate(layer_names)))

    return df


def _plot_grid_hist(
    df_NL: pd.DataFrame,
    nbins: int | None = None,
) -> go.Figure:
    layers = df_NL["Layer"].unique()

    fig = make_subplots(rows=1, cols=len(layers))

    for layer_idx, layer in enumerate(layers):
        df_N = df_NL[df_NL["Layer"] == layer]

        hist = go.Histogram(
            x=df_N["Values"],
            histnorm="percent",
            nbinsx=nbins,
            name=f"{layer}",
        )
        fig.add_trace(hist, row=1, col=layer_idx + 1)

    return fig


def _plot_overlay_hist(
    df_NL: pd.DataFrame,
    nbins: int | None = None,
) -> go.Figure:
    fig = px.histogram(
        df_NL,
        x="Values",
        color="Layer",
        marginal="rug",
        histnorm="percent",
        nbins=nbins,
        opacity=0.5,
        barmode="overlay",
    )

    return fig


def plot_norms_hists(
    norms_NML: torch.Tensor,
    model_names: list[str] | None = None,
    layer_names: list[int] | list[str] | None = None,
    k_iqr: float | None = None,
    overlay: bool = True,
    log: bool = True,
    nbins: int | None = None,
    title: str = "Residual Stream Magnitude",
    xaxis_title: str = "Norm",
    yaxis_title: str = "Percentage",
) -> list[go.Figure]:
    df = _build_df(norms_NML, model_names=model_names, layer_names=layer_names, k_iqr=k_iqr)

    if log:
        df["Values"] = torch.log10(torch.tensor(df["Values"]))

    fig_list = []
    n_models = norms_NML.shape[1]

    for model_name in model_names if model_names else range(n_models):
        model_df = df.loc[df["Model"] == model_name]

        fig = _plot_overlay_hist(model_df, nbins=nbins) if overlay else _plot_grid_hist(model_df, nbins=nbins)

        fig.update_layout(
            legend_title_text="Layer",
            title=f"{title} ({model_name})" if model_names else title,
            xaxis_title=f"{xaxis_title} (log10)" if log else xaxis_title,
            yaxis_title=yaxis_title,
        )

        fig_list.append(fig)

    return fig_list
