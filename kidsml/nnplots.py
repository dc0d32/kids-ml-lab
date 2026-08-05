"""Pictures for the neural-network chapters.

These helpers keep the pages and notebooks drawing neurons the same way: one set of
colours, one diagram language, and small grids so every picture stays laptop-sized.
"""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from kidsml.nn_numpy import ACTIVATIONS, MLP, Neuron, mse
from kidsml.plots import ACCENT, COOL, MUTED, WARM, decision_boundary, draw_line, scatter_2d

HIDDEN_COLOURS = [ACCENT, '#8B5CF6', '#F59E0B', '#14B8A6', '#EC4899', '#6366F1', '#84CC16', '#F97316']


def _round_box(ax, xy, width, height, text, face='#F8FAFC', edge=MUTED):
    box = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.05,rounding_size=0.08',
        facecolor=face, edgecolor=edge, linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha='center', va='center', fontsize=11)
    return box


def _arrow(ax, start, end, label=None, colour=MUTED):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=16, linewidth=1.7, color=colour)
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.08, label, ha='center', va='bottom', fontsize=10, color=colour)


def neuron_diagram(weights=(2, -1), bias=0.5, activation='sigmoid'):
    """Draw the standard one-neuron picture: inputs, weighted sum, squish, output."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    ys = [2.9, 1.45]
    labels = ['$x_1$', '$x_2$']
    for i, y in enumerate(ys):
        circ = Circle((1.0, y), 0.35, facecolor='#DBEAFE', edgecolor=COOL, linewidth=1.8)
        ax.add_patch(circ)
        ax.text(1.0, y, labels[i], ha='center', va='center', fontsize=12)
        _arrow(ax, (1.35, y), (3.7, 2.2), f'w{i + 1}={weights[i]:g}')

    ax.text(2.6, 0.55, f'bias b = {bias:g}', ha='center', va='center', fontsize=10, color=MUTED)
    _arrow(ax, (2.9, 0.75), (3.8, 1.95), None)

    sum_circle = Circle((4.2, 2.2), 0.52, facecolor='#DCFCE7', edgecolor=ACCENT, linewidth=2.0)
    ax.add_patch(sum_circle)
    ax.text(4.2, 2.2, 'add\nup', ha='center', va='center', fontsize=11)
    ax.text(4.2, 3.05, '$z = w_1x_1 + w_2x_2 + b$', ha='center', fontsize=12)

    _arrow(ax, (4.72, 2.2), (5.8, 2.2), 'z', ACCENT)
    _round_box(ax, (5.9, 1.65), 1.55, 1.1, f'{activation}\nsquish', face='#FEF3C7', edge='#F59E0B')
    _arrow(ax, (7.45, 2.2), (8.55, 2.2), '0..1', WARM)

    out = Circle((9.05, 2.2), 0.42, facecolor='#FEE2E2', edgecolor=WARM, linewidth=1.8)
    ax.add_patch(out)
    ax.text(9.05, 2.2, 'out', ha='center', va='center', fontsize=11)
    fig.tight_layout()
    return fig


def network_diagram(sizes=(2, 3, 1), activation='tanh'):
    """Draw a small circles-and-arrows network without hiding the layer sizes."""
    sizes = list(sizes)
    fig, ax = plt.subplots(figsize=(2.7 * len(sizes), 4.5))
    ax.axis('off')
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-0.5, max(sizes) - 0.5)

    positions = []
    for layer, size in enumerate(sizes):
        top = (max(sizes) - 1) / 2
        ys = np.linspace(top + (size - 1) / 2, top - (size - 1) / 2, size)
        layer_pos = []
        for j, y in enumerate(ys):
            if layer == 0:
                face, edge = '#DBEAFE', COOL
                text = f'x{j + 1}'
            elif layer == len(sizes) - 1:
                face, edge = '#FEE2E2', WARM
                text = 'out' if size == 1 else str(j + 1)
            else:
                face, edge = '#DCFCE7', HIDDEN_COLOURS[(j) % len(HIDDEN_COLOURS)]
                text = f'h{j + 1}'
            circ = Circle((layer, y), 0.24, facecolor=face, edgecolor=edge, linewidth=1.8, zorder=3)
            ax.add_patch(circ)
            ax.text(layer, y, text, ha='center', va='center', fontsize=9, zorder=4)
            layer_pos.append((layer, y))
        positions.append(layer_pos)

    for layer in range(len(sizes) - 1):
        for start in positions[layer]:
            for end in positions[layer + 1]:
                ax.plot([start[0] + 0.24, end[0] - 0.24], [start[1], end[1]], color='#CBD5E1', linewidth=1.1, zorder=1)

    labels = ['inputs']
    for layer in range(1, len(sizes) - 1):
        labels.append(f'hidden layer {layer}\n{activation}')
    labels.append('output\nsigmoid')
    for layer, label in enumerate(labels):
        ax.text(layer, -0.35, label, ha='center', va='top', fontsize=10, color=MUTED)
    fig.tight_layout()
    return fig


def neuron_surface_figure(neuron: Neuron, X, steps: int = 50, title: str = 'Neuron output surface'):
    """A rotatable 3D surface showing one neuron's output over the input plane."""
    X = np.asarray(X, dtype=float)
    pad = 0.6
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, steps)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, steps)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = neuron.forward(grid).reshape(xx.shape)

    fig = go.Figure(data=[go.Surface(x=xx, y=yy, z=zz, colorscale='RdBu_r', showscale=False, opacity=0.95)])
    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='output'),
    )
    return fig


def hidden_lines(ax, model: MLP, labels: bool = True):
    """Draw each first-hidden-layer neuron's own line on an existing 2D plot."""
    W = model.Ws[0]
    b = model.bs[0]
    for j in range(W.shape[1]):
        label = f'hidden {j + 1}' if labels else None
        draw_line(W[0, j], W[1, j], b[j], ax=ax, color=HIDDEN_COLOURS[j % len(HIDDEN_COLOURS)], label=label, linewidth=2.0)
    if labels:
        ax.legend(loc='best', fontsize=8)
    return ax


def hidden_surfaces_figure(model: MLP, X, steps: int = 75):
    """Small panels showing the first hidden layer's outputs as ramps over the plane."""
    X = np.asarray(X, dtype=float)
    n_hidden = model.Ws[0].shape[1]
    fig, axes = plt.subplots(1, n_hidden, figsize=(4.2 * n_hidden, 3.6), squeeze=False)
    pad = 0.6
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, steps)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, steps)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    hidden = model.hidden_outputs(grid).reshape(xx.shape + (n_hidden,))

    for j, ax in enumerate(axes[0]):
        ax.contourf(xx, yy, hidden[:, :, j], levels=20, cmap='RdBu_r', alpha=0.9)
        scatter_2d(X, None, ax=ax, size=16, alpha=0.35, labels=False)
        draw_line(model.Ws[0][0, j], model.Ws[0][1, j], model.bs[0][j], ax=ax, color=HIDDEN_COLOURS[j % len(HIDDEN_COLOURS)])
        ax.set_title(f'hidden neuron {j + 1}')
    fig.tight_layout()
    return fig


def mlp_snapshot_training(sizes, X, y, lr=0.8, epochs=1200, every=100, activation='tanh', seed=0, weight_decay=0.0):
    """Train an MLP and keep lightweight copies of its weights along the way."""
    model = MLP(sizes, activation=activation, seed=seed)
    snapshots = []
    for step in range(epochs + 1):
        if step % every == 0:
            snapshots.append({
                'step': step,
                'Ws': [W.copy() for W in model.Ws],
                'bs': [b.copy() for b in model.bs],
                'loss': mse(model.forward(X), np.asarray(y).reshape(-1, 1)),
            })
        if step < epochs:
            model.step(X, y, lr=lr, weight_decay=weight_decay)
    return snapshots


def model_from_snapshot(sizes, snapshot, activation='tanh', seed=0):
    """Rebuild an MLP from a snapshot made by :func:`mlp_snapshot_training`."""
    model = MLP(sizes, activation=activation, seed=seed)
    model.Ws = [W.copy() for W in snapshot['Ws']]
    model.bs = [b.copy() for b in snapshot['bs']]
    return model


def copy_mlp(model: MLP) -> MLP:
    """Return a separate MLP with the same weights."""
    return copy.deepcopy(model)


def boundary_with_hidden(model: MLP, X, y=None, ax=None, title='Final boundary plus hidden lines', steps: int = 220):
    """Decision boundary plus the first hidden layer's own straight lines."""
    ax = ax or plt.gca()
    decision_boundary(lambda G: model.predict_proba(G), X, y, ax=ax, steps=steps, shade_confidence=True, title=title)
    hidden_lines(ax, model)
    return ax
