import os
import re

import plotly_express as px
import plotly.graph_objects as go


def list_scatter(
    dfs: dict,
    stage: str,
    metric: str,
    experiment_name: str,
    color: str,
    showlegend: bool,
    custom_index=None,
    plot_best: bool = True,
):
    opct_std = 0.15
    opct_mean = 1
    opct_best = 1
    index = dfs[stage].index if custom_index is None else custom_index
    upper_bound = dfs[stage][f"{metric}_mean"] + dfs[stage][f"{metric}_std"]
    lower_bound = dfs[stage][f"{metric}_mean"] - dfs[stage][f"{metric}_std"]
    traces = [
        go.Scatter(
            name="Upper Bound",
            x=index,
            y=upper_bound,
            mode="lines",
            # marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
        ),
        go.Scatter(
            name="Standard Deviation",
            x=index,
            y=lower_bound,
            mode="none",
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [opct_std])}",
            fill="tonexty",
            showlegend=False,
        ),
        go.Scatter(
            name=f"{experiment_name}",
            x=index,
            y=dfs[stage][f"{metric}_mean"],
            mode="lines",
            line=dict(
                color=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [opct_mean])}",
                width=2,
            ),
            showlegend=showlegend,
        ),
        go.Scatter(
            name=f"{experiment_name}",
            x=index,
            y=dfs[stage][f"{metric}_best"],
            mode="lines",
            line=dict(
                color=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [opct_best])}",
                dash="dot",
                width=1,
            ),
            showlegend=False,
        ),
    ]
    if not plot_best:
        traces.pop(3)
    return traces


def update_layout(
    fig,
    stage: str,
    metric: str,
    showlegend: bool,
    yanchor: str = "bottom",
    custom_width: int = None,
    custom_height: int = None,
):
    fig.update_layout(
        yaxis_title=f"{stage.upper()} {metric.replace('_', ' ').upper()}",
        xaxis_title="Epochs",
        title="",
        hovermode="x",
        width=custom_width
        if custom_width is not None
        else (500 if showlegend else 300),
        height=custom_height if custom_height is not None else 300,
        # plot_bgcolor="rgba(0,0,0,0)",
        # paper_bgcolor="rgba(0,0,0,0)",
        # make legend font smaller
        # anchor legend to the top right
        legend=dict(
            y=0.1 if yanchor == "bottom" else 0.99,
            traceorder="normal",
            yanchor=yanchor,
            xanchor="right",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        template="simple_white",
        font_family="Serif",
    )


def condition_fn(regex: str, name: str):
    return re.search(regex, name) is not None


def save_images(
    fig: go.Figure,
    stage: str,
    metric: str,
    exp_dir: str,
    to_search: str,
    scale: int,
    format: tuple = ("png", "svg", "pdf", "eps", "emf"),
):
    for c in format:
        path = f"{exp_dir}/{c}/{to_search.replace('.*', '/')}"
        if not os.path.exists(path):
            os.makedirs(path)
        if c == "emf":
            fig.write_json(f"{path}/{stage}_{metric}.{c}")
        else:
            fig.write_image(f"{path}/{stage}_{metric}.{c}", scale=scale)


def transform_legend(legend: str, to_remove: list = []):
    legend = legend.replace("Block Branch", "BB")
    legend = legend.replace("Freeze Evolved", "FEB")
    legend = legend.replace("Freeze Output Layers", "FEOL")
    # replace min lr
    legend = re.sub(
        r", Min Lr = (\d+.\d+); Max Lr = (\d+.\d+)", r" (η ∈ [\1, \2])", legend
    )
    # replace with regex
    legend = re.sub(r", Epochs = *\d+", "", legend)
    for s in to_remove:
        legend = re.sub(rf"{s} = False, ", "", legend)
        legend = re.sub(rf"{s} = True, ", "", legend)
    legend = legend.replace("Lrf = ", "LRF = ")
    legend = legend.replace(", LRF", "<br>LRF")
    legend = legend.replace("LRF = False", "LRF = False (η = 0.05)")

    # replace "Loss Hparams = {'Label Smoothing': 0.05}, Epochs = 20" with LS = 0.1
    legend = re.sub(r"Loss Hparams = {'Label Smoothing': (\d+.\d+)}", r"𝛼 = \1", legend)
    legend = re.sub(r"Loss Hparams = {'Gamma': (\d+.\d+)}", r"γ = \1", legend)
    legend = re.sub(r"Loss Hparams = {'Gamma': (\d+)}", r"γ = \1", legend)
    legend = re.sub(r"Loss Hparams = {'Epsilon': (\d+)}", r"ϵ = \1", legend)
    legend = re.sub(r"Generations", r"Gen", legend)
    legend = re.sub(r"Population", r"Pop", legend)
    legend = re.sub(r", Use Lr Finder = False", r"", legend)

    # legend = legend.replace("= False", "False").replace("= True", "True")
    # remove to_remove
    # legend = legend.replace("e, ", "e<br>")
    # legend = "Baseline"

    return legend
