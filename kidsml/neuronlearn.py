"""Helpers for Chapter 14 · How a Neuron Learns.

The star of the chapter is the picture of the weights walking downhill. To make that
picture honest we have to be careful about two things, and both live here so the app page
and the notebook draw the exact same thing:

* The loss the path walks on must be the loss drawn behind it. We hold the bias ``b``
  fixed, so the whole story is about two numbers ``w1`` and ``w2`` and the loss really is a
  2D surface we can draw.
* The problem has to have a *bottom* you can reach. Two clean, far-apart blobs have no
  bottom — the weights just grow forever and the walk is a boring straight line. So we use
  two blobs that overlap, which gives the loss a real valley with a lowest point sitting at
  finite weights. Now a small step crawls, a good step settles in, and a big step flies
  past — which is the entire lesson.
"""

from __future__ import annotations

import numpy as np

from kidsml.nn_numpy import sigmoid, sigmoid_slope
from kidsml.plots import ACCENT, COUNT_CMAP, GRIDLINE, INK, PANEL

# Bright markers for the walk, kept here so they read on the pure-black page no matter what
# the shared palette currently holds: the moving path is green, the start is white, where it
# stopped is amber, and the lowest point is pink.
AMBER = "#FBBF24"
PINK = "#F472B6"

# The bias we hold still while the two weights learn, so the loss is a flat 2D map.
FIXED_B = 0.0
# How far apart the walk grid is drawn, in weight units.
_GRID = 3.2


def valley_data(n: int = 160, seed: int = 3):
    """Two overlapping blobs whose loss valley has a real lowest point.

    ``x1`` carries most of the class signal and is spread wide, so the loss changes fast
    when ``w1`` changes — a steep wall. ``x2`` is spread narrow, so ``w2`` changes the loss
    slowly — a shallow floor. Steep one way and shallow the other is exactly what makes a
    valley you can walk *along*.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    blue = np.column_stack([rng.normal(-1.0, 1.5, half), rng.normal(0.0, 0.45, half)])
    red = np.column_stack([rng.normal(1.0, 1.5, half), rng.normal(0.0, 0.45, half)])
    X = np.vstack([blue, red]) * 2.8
    y = np.concatenate([np.zeros(half), np.ones(half)])
    return X, y


def _loss(X, y, w1, w2):
    return float(np.mean((sigmoid(X @ np.array([w1, w2]) + FIXED_B) - y) ** 2))


def loss_grid(X, y, steps: int = 90):
    """A grid of the loss over every ``(w1, w2)`` pair, ready for filled contours.

    Returns ``(W1, W2, Z, w_star)`` where ``w_star`` is the lowest point on the grid.
    """
    axis = np.linspace(-_GRID, _GRID, steps)
    W1, W2 = np.meshgrid(axis, axis)
    Z = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Z[i, j] = _loss(X, y, W1[i, j], W2[i, j])
    k = np.unravel_index(np.argmin(Z), Z.shape)
    return W1, W2, Z, (float(W1[k]), float(W2[k]))


def walk_start(seed: int):
    """Where this walk begins. High up one wall of the valley, so the first steps have
    somewhere to go. The seed only jiggles the start a little, so every walk faces the
    same valley."""
    rng = np.random.default_rng(seed)
    return np.array([-0.5, 2.3]) + rng.normal(0, 0.25, size=2)


def weight_walk(X, y, lr: float, seed: int, steps: int = 90):
    """Take ``steps`` downhill steps on the two weights, holding the bias fixed.

    Returns the array of ``(w1, w2)`` after every step, including the start, so the path
    can be drawn dot by dot.
    """
    w = walk_start(seed)
    n = len(y)
    path = [w.copy()]
    for _ in range(steps):
        z = X @ w + FIXED_B
        out = sigmoid(z)
        dL_dz = 2.0 * (out - y) / n * sigmoid_slope(z)
        w = w - lr * (X.T @ dL_dz)
        path.append(w.copy())
    return np.array(path)


def draw_weight_walk(ax, lr: float, seed: int) -> None:
    """Draw the loss valley and one downhill walk across it onto ``ax``.

    Both the app page and the notebook call this, so the two never drift apart.
    """
    X, y = valley_data()
    W1, W2, Z, star = loss_grid(X, y)
    path = weight_walk(X, y, lr, seed)

    ax.contourf(W1, W2, Z, levels=18, cmap=COUNT_CMAP, alpha=0.85)
    ax.contour(W1, W2, Z, levels=8, colors=[GRIDLINE], linewidths=0.6)
    ax.plot(path[:, 0], path[:, 1], color=ACCENT, lw=1.4, marker="o", markersize=3, zorder=4)
    ax.scatter([path[0, 0]], [path[0, 1]], color=INK, edgecolors=PANEL, s=70, marker="s", zorder=5, label="start")
    ax.scatter([path[-1, 0]], [path[-1, 1]], color=AMBER, s=80, marker="X", zorder=5, label="where it stopped")
    ax.scatter([star[0]], [star[1]], color=PINK, s=150, marker="*", zorder=6, label="lowest loss")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("The weights walk down into the loss valley")
    ax.legend(loc="upper right", fontsize=7, facecolor="#0B0B0D", labelcolor=INK)
