# Plotly
from optparse import Option
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from typing import Literal, Optional, Union

import numpy as np
import networkx as nx
from pytorch_lightning import LightningModule

import torch as torch
from torch import Tensor
from torch.nn import Module
from modules.linear.block import LinearBlock

from pytorch_lightning import Trainer, LightningModule
import torch.nn as nn

from modules.model import EvoModel
import modules.param_init as param_init


def plot_graph(
    model: EvoModel,
    line_opacity: float = 0.05,
    skip_input_layer: bool = True,
    basic: bool = False,
    renderer: Literal["notebook", "browser"] = "browser",
    layout: Literal["random", "layered", "circular", "kamada_kawai"] = "layered",
    height: int = 800,
    plot_trace: bool = True,
    template: str = "plotly",
    save_html: bool = False,
):
    """Plot Neural Network Structure

    Args:
        model (LightningModule | Module): Model to plot
        renderer (["notebook", "browser"], optional): Where to plot. Defaults to "browser".
    """
    adj_matrix = _adj_matrix_create(model)
    if skip_input_layer:
        adj_matrix = adj_matrix[model.structure[0] :, model.structure[0] :]
    adj_np = ((adj_matrix != 0) * 1).numpy()
    graph = nx.from_numpy_matrix(adj_np)
    neuron_layer = []
    for i, n in enumerate((model.structure if not skip_input_layer else model.structure[1:])):
        neuron_layer += [i] * n
    layer_dict = {i: layer for i, layer in enumerate(neuron_layer)}
    weight_dict = {(i, j): weight for i, weights in enumerate(adj_matrix.numpy()) for j, weight in enumerate(weights)}
    nx.set_edge_attributes(graph, weight_dict, "weight")
    nx.set_node_attributes(graph, layer_dict, "layer")

    if layout == "layered":
        pos = nx.multipartite_layout(graph, "layer")
    elif layout == "random":
        pos = nx.random_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        raise ValueError(
            f"""Layout {layout} not supported. 
        Select from ['layered', 'random', 'kamada_kawai', 'circular']"""
        )
    fig = _plotly_graph(
        graph, pos, line_opacity, renderer, height=height, plot_trace=plot_trace, template=template, basic=basic,
    )
    if save_html:
        fig.write_html("graph.html")
    fig.show(renderer=renderer)
    return adj_matrix, adj_np, graph


def _adj_matrixs(adj_matrix: Tensor, module):
    if hasattr(module, "weight_mask"):
        w_ = module.weight.detach() * module.weight_mask.detach()
    else:
        w_ = module.weight.detach()
    old_n_neurons = adj_matrix.shape[0]
    new_n_neurons = adj_matrix.shape[0] + w_.shape[0]
    new_adj = torch.zeros((new_n_neurons, new_n_neurons))
    new_adj[:old_n_neurons, :old_n_neurons] = adj_matrix
    new_adj[old_n_neurons:, : w_.shape[1]] = w_
    return new_adj


def _adj_matrix_add_block(adj_matrix: Tensor, block: LinearBlock):
    if block.is_frozen:
        adj_matrix = _adj_matrixs(adj_matrix, block.exp_f)
    if block.has_unfrozen_modules:
        adj_matrix = _adj_matrixs(adj_matrix, block.exp_uf)
    new_size = adj_matrix.shape[0] + block.num_outputs
    new_adj = torch.zeros((new_size, new_size,))
    new_adj[: adj_matrix.shape[0], : adj_matrix.shape[0]] = adj_matrix
    new_adj[adj_matrix.shape[0] :, (r := block.in_features) : (r := r + block.num_outputs),] = torch.eye(
        block.num_outputs
    )
    if block.is_frozen:
        new_adj[adj_matrix.shape[0] :, r : (r := r + block.exp_f.out_features)] = (
            block.agg_f.weight.data.detach() * block.agg_f.weight_mask.detach()
        )
    if block.has_unfrozen_modules:
        new_adj[adj_matrix.shape[0] :, r : (r := r + block.exp_uf.out_features)] = (
            block.agg_uf.weight.data.detach() * block.agg_uf.weight_mask.detach()
        )
    return new_adj


def _adj_matrix_create(model: LightningModule):
    in_features = model.main_module.in_features
    adj = torch.zeros((in_features, in_features))
    adj = _adj_matrixs(adj, model.main_module)
    if model.n_blocks > 0:
        for i in range(model.n_blocks):
            adj = _adj_matrix_add_block(adj, model.get_submodule(f"block{i}"))
    return adj + adj.T


def _plotly_graph(
    graph: nx.Graph,
    pos: dict,
    line_opacity: float,
    renderer: Literal["notebook", "browser"],
    height: int,
    plot_trace: bool = True,
    template: str = "plotly",
    basic: bool = False,
):
    edge_x = []
    edge_y = []
    edge_traces = []

    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        if not basic:
            edge_traces.append(
                go.Scattergl(
                    x=[x0, x1],
                    y=[y0, y1],
                    line=dict(
                        width=(abs(edge[2]["weight"]) + 1) * 2,
                        color=(
                            "lightgrey" if edge[2]["weight"] == 1 else "sienna" if edge[2]["weight"] < 0 else "darkcyan"
                        ),
                    ),
                    opacity=line_opacity,
                    # hoverinfo="none",
                    mode="lines",
                )
            )

    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        opacity=line_opacity,
        # hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scattergl(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale="Greys",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right",),
            line_width=0.5,
        ),
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append("# of connections: " + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=([edge_trace] if basic else edge_traces) + [node_trace] if plot_trace else [node_trace],
        layout=go.Layout(
            # title='<br>Neural Network Graph',
            titlefont_size=16,
            showlegend=False,
            height=height,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            # annotations=[ dict(
            #     text="Txt",
            #     showarrow=False,
            #     xref="paper", yref="paper",
            #     x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    # Remove background color from graph
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # add dark template
        template=template,
    )
    return fig




def plot_weights(
    tensor: Tensor,
    bin_size: float = 0.1,
    c: float = 1.0,
    renderer: Literal["notebook", "browser"] = "notebook",
    template: str = "plotly",
):
    valid_weights = tensor.detach().flatten().numpy()
    fig = ff.create_distplot([valid_weights], ["valid_weights"], colors=["#0072B2"], bin_size=bin_size, show_rug=False,)

    fig.layout.title = f"sparsity = {c:.4f} || std = {np.std(valid_weights):.4f}"

    fig.show(renderer="notebook")


def rd_mean_std_plot(
    means: list,
    stds: list,
    maxs: Optional[list] = None,
    names: Optional[list] = None,
    stabilizers: Optional[list] = None,
    dark_mode=False,
    height=600,
    renderer: Literal["notebook", "browser"] = "notebook",
    title: str = "",
):
    colors = px.colors.qualitative.Plotly
    dark_mode_dict = dict(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_font=dict(size=10, color="white"),
        legend_bgcolor="rgba(0,0,0,0)",
        legend_bordercolor="rgba(0,0,0,0)",
    )
    names = names or [f"{i}" for i in range(len(means))]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=list(range(len(x))) + list(range(len(x)))[::-1],
                y=[mean + std for mean, std in zip(means[i], stds[i])]
                + [mean - std for mean, std in zip(means[i], stds[i])][::-1],
                name=names[i],
                mode="lines",
                fill="toself",
                fillcolor=colors[i % len(colors)],
                opacity=0.2,
                # line_width=0,
                showlegend=False,
                hoverinfo="skip",
                legendgroup=i,
            )
            for i, x in enumerate(stds)
        ]
        + [
            go.Scatter(
                x=list(range(len(x))),
                y=x,
                name=names[i],
                marker_color=colors[i % len(colors)],
                mode="lines",
                # add hover info of mean and std
                hovertemplate="<b>Mean</b>: %{y:.4f}<br><b>Std</b>: %{text}",
                text=[f"{round(stds[i][j],4)}" for j in range(len(x))],
                legendgroup=i,
            )
            for i, x in enumerate(means)
        ]
    )
    # change x axis name to layer number
    fig.update_xaxes(title_text="Layer")
    # change variable names
    fig.update_yaxes(title_text=title)
    # Change legend title
    fig.update_layout(
        # add title
        title_text=title,
        legend_y=-0.2,
        legend_x=0.5,
        legend_title_text="",
        legend_orientation="h",
        legend_xanchor="center",
        hovermode="x",
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
    )
    if dark_mode:
        fig.update_layout(dark_mode_dict)
    fig.show(renderer=renderer)

    if maxs is not None:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(range(len(x))),
                    y=x,
                    name=names[i],
                    marker_color=colors[i % len(colors)],
                    mode="lines",
                    # add hover info of mean and std
                    hovertemplate="<b>Max</b>: %{y:.4f}",
                )
                for i, x in enumerate(maxs)
            ]
        )
        # change x axis name to layer number
        fig.update_xaxes(title_text="Layer")
        # change variable names
        fig.update_yaxes(title_text="Max Values")
        # Change legend title
        fig.update_layout(
            legend_y=-0.2,
            legend_x=0.5,
            legend_title_text="",
            legend_orientation="h",
            legend_xanchor="center",
            hovermode="x",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if dark_mode:
            fig.update_layout(dark_mode_dict)
        fig.show(renderer=renderer)

    if stabilizers is not None:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(range(len(x))),
                    y=x,
                    name=names[i],
                    marker_color=colors[i % len(colors)],
                    mode="lines",
                    # add hover info of mean and std
                    hovertemplate="%{y:.4f}",
                )
                for i, x in enumerate(stabilizers)
            ]
        )
        # change x axis name to layer number
        fig.update_xaxes(title_text="Layer")
        # change variable names
        fig.update_yaxes(title_text="Stabilizer Values")
        # Change legend title
        fig.update_layout(
            title_text=title,
            legend_y=-0.2,
            legend_x=0.5,
            legend_title_text="",
            legend_orientation="h",
            legend_xanchor="center",
            hovermode="x",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if dark_mode:
            fig.update_layout(dark_mode_dict)
        fig.show(renderer=renderer)


def plot_all_params(model: LightningModule, inputs: Optional[Tensor], height: int = 600, n_epochs: Optional[int] = None):
    xs_randn = param_init.xs_randn(model, inputs)
    y_randn, layers_y_randn = model(xs_randn[0][0])
    xs_mean = []
    xs_std = []

    for i, block in enumerate(xs_randn):
        if i != 0:
            for mod in block:
                xs_mean.append(mod.mean().item())
                xs_std.append(mod.std().item())
            xs_mean.append(layers_y_randn[i-1].mean().item())
            xs_std.append(layers_y_randn[i-1].std().item())

    xs_mean.append(y_randn.mean().item())
    xs_std.append(y_randn.std().item())
    rd_mean_std_plot(
        [xs_mean], [xs_std], dark_mode=False, height=height, renderer="browser", title=f"random init / EPOCH {n_epochs}"
    )

    layer_mean = []
    layer_std = []
    counter = 0

    for name, param in model.named_parameters():
        if "layer_" in name and ".regularization." not in name and "layer_out" not in name:
            layer_mean.append(param.data.mean().item())
            layer_std.append(param.data.std().item())
            counter += 1
    rd_mean_std_plot(
        [layer_mean], [layer_std], dark_mode=False, height=height, renderer="browser", title=f"layer weights / EPOCH {n_epochs}"
    )

    out_mean = []
    out_std = []
    counter = 0

    for name, param in model.named_parameters():
        if "layer_out" in name and ".bn." not in name:
            out_mean.append(param.data.mean().item())
            out_std.append(param.data.std().item())
            counter += 1
    rd_mean_std_plot(
        [out_mean], [out_std], dark_mode=False, height=height, renderer="browser", title=f"layer_out weights / EPOCH {n_epochs}"
    )
