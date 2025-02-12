import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
import torch
from plotly.subplots import make_subplots  # type: ignore

from model_diffing.analysis import metrics


def plot_relative_norms(vectors_a_NF: torch.Tensor, vectors_b_NF: torch.Tensor, title: str | None = None) -> go.Figure:
    """Plot histogram of relative norms (norm_b / (norm_a + norm_b)).

    Args:
        vectors_a_NF: Tensor of vectors from the first set
        vectors_b_NF: Tensor of vectors from the second set
        title: Optional title for the plot

    Returns:
        Plotly figure object
    """
    relative_norms_N = metrics.compute_relative_norms_N(vectors_a_NF, vectors_b_NF)

    return relative_norms_hist(relative_norms_N, title=title)


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
    values_NMP: torch.Tensor,
    model_names: list[str] | None = None,
    hookpoint_names: list[int] | list[str] | None = None,
    k_iqr: float | None = None,
) -> pd.DataFrame:
    df_list = []

    for model_idx in range(values_NMP.shape[1]):
        for hookpoint_idx in range(values_NMP.shape[2]):
            values = values_NMP[:, model_idx, hookpoint_idx]

            if k_iqr:
                outliers = metrics.get_IQR_outliers_mask(values.unsqueeze(1), k_iqr=k_iqr).squeeze()
                values = values[~outliers]

            df_list.append(
                pd.DataFrame(
                    {
                        "Values": values.cpu().numpy(),
                        "hookpoint": hookpoint_idx,
                        "Model": model_idx,
                    }
                )
            )

    df = pd.concat(df_list, ignore_index=True)

    # Map model and hookpoint indices to the provided names
    if model_names:
        df["Model"] = df["Model"].replace(dict(enumerate(model_names)))
    if hookpoint_names:
        df["hookpoint"] = df["hookpoint"].replace(dict(enumerate(hookpoint_names)))

    return df


def _plot_grid_hist(
    df: pd.DataFrame,
    nbins: int | None = None,
) -> go.Figure:
    hookpoints = df["hookpoint"].unique()

    fig = make_subplots(rows=1, cols=len(hookpoints))

    for hookpoint_idx, hookpoint in enumerate(hookpoints):
        df_this_hookpoint = df[df["hookpoint"] == hookpoint]

        hist = go.Histogram(
            x=df_this_hookpoint["Values"],
            histnorm="percent",
            nbinsx=nbins,
            name=f"{hookpoint}",
        )
        fig.add_trace(hist, row=1, col=hookpoint_idx + 1)

    return fig


def _plot_overlay_hist(
    df: pd.DataFrame,
    nbins: int | None = None,
) -> go.Figure:
    fig = px.histogram(
        df,
        x="Values",
        color="hookpoint",
        marginal="rug",
        histnorm="percent",
        nbins=nbins,
        opacity=0.5,
        barmode="overlay",
    )

    return fig


def plot_norms_hists(
    norms_NMP: torch.Tensor,
    model_names: list[str] | None = None,
    hookpoint_names: list[int] | list[str] | None = None,
    k_iqr: float | None = None,
    overlay: bool = True,
    log: bool = True,
    nbins: int | None = None,
    title: str = "Residual Stream Magnitude",
    xaxis_title: str = "Norm",
    yaxis_title: str = "Percentage",
) -> list[go.Figure]:
    df = _build_df(norms_NMP, model_names=model_names, hookpoint_names=hookpoint_names, k_iqr=k_iqr)

    if log:
        df["Values"] = torch.log10(torch.tensor(df["Values"]))

    fig_list = []
    n_models = norms_NMP.shape[1]

    for model_name in model_names if model_names else range(n_models):
        model_df = df.loc[df["Model"] == model_name]

        fig = _plot_overlay_hist(model_df, nbins=nbins) if overlay else _plot_grid_hist(model_df, nbins=nbins)

        fig.update_layout(
            legend_title_text="hookpoint",
            title=f"{title} ({model_name})" if model_names else title,
            xaxis_title=f"{xaxis_title} (log10)" if log else xaxis_title,
            yaxis_title=yaxis_title,
        )

        fig_list.append(fig)

    return fig_list
