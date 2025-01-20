import plotly.express as px
import plotly.graph_objs as go
import torch

from model_diffing.analysis import metrics


def plot_relative_norms(vectors_a: torch.Tensor, vectors_b: torch.Tensor, title: str | None = None) -> go.Figure:
    """Plot histogram of relative norms (norm_b / (norm_a + norm_b)).

    Args:
        vectors_a: Tensor of vectors from the first set
        vectors_b: Tensor of vectors from the second set
        title: Optional title for the plot

    Returns:
        Plotly figure object
    """
    relative_norms = metrics.compute_relative_norms(vectors_a, vectors_b)

    fig = px.histogram(
        relative_norms.detach().cpu().numpy(),
        nbins=200,
        labels={"value": "Relative norm"},
        title=title,
    )

    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Number of Latents")
    fig.update_xaxes(tickvals=[0, 0.25, 0.5, 0.75, 1.0], ticktext=["0", "0.25", "0.5", "0.75", "1.0"])
    return fig


def plot_cosine_sim(cosine_sims: torch.Tensor, title: str | None = None) -> go.Figure:
    """Plot histogram of cosine similarities.

    Args:
        cosine_sims: Tensor of cosine similarity values
        title: Optional title for the plot

    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        cosine_sims.detach().cpu().numpy(),
        log_y=True,
        range_x=[-1, 1],
        nbins=100,
        labels={"value": "Cosine similarity"},
        title=title,
    )

    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Number of Latents (log scale)")
    return fig
