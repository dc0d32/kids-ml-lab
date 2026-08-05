"""The house drawing style.

One place decides what red and blue mean, how thick a line is, and what a decision
boundary looks like. Notebooks and the Streamlit app both call these, so a picture
looks the same wherever you meet it.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Two classes, two colours, everywhere in the course. Warm = class 1, cool = class 0.
COOL = "#3B82F6"  # blue  — class 0
WARM = "#EF4444"  # red   — class 1
ACCENT = "#10B981"  # green — the model's own line / prediction
MUTED = "#94A3B8"

POINT_CMAP = ListedColormap([COOL, WARM])
REGION_CMAP = LinearSegmentedColormap.from_list("kidsml", ["#BFDBFE", "#FFFFFF", "#FECACA"])

CLASS_NAMES = ("blue", "red")


def use_house_style() -> None:
    """Apply the course's matplotlib defaults. Call once at the top of a notebook."""
    mpl.rcParams.update(
        {
            "figure.figsize": (6.0, 5.0),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "font.size": 11,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "scatter.edgecolors": "white",
        }
    )


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------


def scatter_2d(X, y=None, ax=None, size: int = 45, alpha: float = 0.95, labels: bool = True):
    """Draw a 2D dataset. ``y`` may be None (then everything is grey)."""
    ax = ax or plt.gca()
    if y is None:
        ax.scatter(X[:, 0], X[:, 1], s=size, c=MUTED, edgecolors="white", linewidths=0.8, alpha=alpha)
    else:
        y = np.asarray(y)
        for cls, colour, name in ((0, COOL, CLASS_NAMES[0]), (1, WARM, CLASS_NAMES[1])):
            m = y == cls
            if m.any():
                ax.scatter(
                    X[m, 0], X[m, 1], s=size, c=colour, edgecolors="white",
                    linewidths=0.8, alpha=alpha, label=name if labels else None, zorder=3,
                )
        if labels:
            ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("feature 1  ($x_1$)")
    ax.set_ylabel("feature 2  ($x_2$)")
    return ax


def _grid_for(X, pad: float = 0.6, steps: int = 300):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    return xx, yy


def decision_boundary(
    predict,
    X,
    y=None,
    ax=None,
    steps: int = 300,
    shade_confidence: bool = True,
    title: str | None = None,
):
    """Colour the whole plane by what ``predict`` says, then drop the data on top.

    ``predict`` takes an ``(n, 2)`` array and returns either a hard 0/1 label or a
    score/probability. If it returns a score and ``shade_confidence`` is on, the colour
    fades near the boundary — that fade *is* the model's uncertainty.
    """
    ax = ax or plt.gca()
    xx, yy = _grid_for(X, steps=steps)
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = np.asarray(predict(grid), dtype=float).ravel().reshape(xx.shape)

    looks_hard = np.isin(np.unique(z), (0.0, 1.0)).all()
    if looks_hard or not shade_confidence:
        ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], cmap=REGION_CMAP, alpha=0.85)
    else:
        lo, hi = float(np.min(z)), float(np.max(z))
        mid = 0.5 if (lo >= 0.0 and hi <= 1.0) else 0.0
        span = max(abs(lo - mid), abs(hi - mid), 1e-9)
        ax.contourf(xx, yy, z, levels=np.linspace(mid - span, mid + span, 25), cmap=REGION_CMAP, alpha=0.85)
        ax.contour(xx, yy, z, levels=[mid], colors=[ACCENT], linewidths=2.5)

    if y is not None:
        scatter_2d(X, y, ax=ax)
    else:
        scatter_2d(X, None, ax=ax)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    if title:
        ax.set_title(title)
    return ax


def draw_line(w1: float, w2: float, b: float, ax=None, color: str = ACCENT, label: str | None = None, **kw):
    """Draw the line ``w1*x1 + w2*x2 + b = 0`` across the current axes."""
    ax = ax or plt.gca()
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    if abs(w2) > 1e-9:
        xs = np.linspace(x_lo, x_hi, 2)
        ys = -(w1 * xs + b) / w2
    elif abs(w1) > 1e-9:
        ys = np.linspace(y_lo, y_hi, 2)
        xs = np.full_like(ys, -b / w1)
    else:
        return ax
    ax.plot(xs, ys, color=color, label=label, zorder=4, **kw)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    return ax


# ---------------------------------------------------------------------------
# Regression pictures
# ---------------------------------------------------------------------------


def regression_fit(x, y, w: float, b: float, ax=None, show_errors: bool = True, show_squares: bool = False):
    """Points, your line, and a stick for every mistake.

    With ``show_squares`` each mistake also gets drawn as an actual square, because
    "squared error" means exactly that: the area of that square.
    """
    ax = ax or plt.gca()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = w * x + b

    if show_errors:
        for xi, yi, pi in zip(x, y, pred):
            ax.plot([xi, xi], [yi, pi], color=MUTED, linewidth=1.6, zorder=1)
    if show_squares:
        for xi, yi, pi in zip(x, y, pred):
            side = yi - pi
            ax.add_patch(
                mpl.patches.Rectangle(
                    (xi, min(yi, pi)), abs(side), abs(side),
                    facecolor=WARM, alpha=0.18, edgecolor=WARM, linewidth=1.0, zorder=1,
                )
            )
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, w * xs + b, color=ACCENT, zorder=2, label=f"y = {w:.2f}·x + {b:.2f}")
    ax.scatter(x, y, s=55, c=COOL, edgecolors="white", linewidths=0.9, zorder=3, label="the real data")
    ax.legend(loc="best", fontsize=9)
    return ax


def loss_curve(losses, ax=None, title: str = "How wrong we are, over time", ylabel: str = "loss"):
    """The 'am I getting better?' plot."""
    ax = ax or plt.gca()
    ax.plot(np.asarray(losses), color=ACCENT)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def loss_surface(x, y, w_range=(-1, 7), b_range=(-10, 20), steps: int = 120):
    """Grid of total squared error over ``(w, b)`` — the valley gradient descent walks down.

    Returns ``(W, B, Z)`` ready for ``contourf`` or a plotly 3D surface.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ws = np.linspace(*w_range, steps)
    bs = np.linspace(*b_range, steps)
    W, B = np.meshgrid(ws, bs)
    pred = W[..., None] * x + B[..., None]
    Z = ((pred - y) ** 2).mean(axis=-1)
    return W, B, Z


# ---------------------------------------------------------------------------
# Images and grids
# ---------------------------------------------------------------------------


def show_image(img, ax=None, title: str | None = None, cmap: str = "gray_r", numbers: bool = False):
    """Show a small image. With ``numbers=True`` the pixel values are printed on top."""
    ax = ax or plt.gca()
    img = np.asarray(img, dtype=float)
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if numbers:
        lo, hi = img.min(), img.max()
        for (r, c), v in np.ndenumerate(img):
            shade = (v - lo) / (hi - lo + 1e-9)
            ax.text(
                c, r, f"{v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if shade > 0.55 else "#111827",
            )
    if title:
        ax.set_title(title)
    return ax


def image_strip(images, titles=None, cmap: str = "gray_r", width: float = 1.5):
    """A row of small images. Handy for 'here are 8 examples' moments."""
    images = list(images)
    fig, axes = plt.subplots(1, len(images), figsize=(width * len(images), width + 0.4))
    axes = np.atleast_1d(axes)
    for i, (ax, img) in enumerate(zip(axes, images)):
        show_image(img, ax=ax, cmap=cmap, title=None if titles is None else str(titles[i]))
    fig.tight_layout()
    return fig, axes


def confusion_grid(cm, labels=None, ax=None, title: str = "What gets mixed up with what"):
    """A confusion matrix you can actually read: counts printed in the cells."""
    ax = ax or plt.gca()
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues")
    n = cm.shape[0]
    labels = list(range(n)) if labels is None else list(labels)
    ax.set_xticks(range(n), labels, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("what the model guessed")
    ax.set_ylabel("the true answer")
    ax.set_title(title)
    ax.grid(False)
    big = cm.max() if cm.max() else 1
    for (r, c), v in np.ndenumerate(cm):
        if v:
            ax.text(c, r, str(int(v)), ha="center", va="center", fontsize=8,
                    color="white" if v > big * 0.55 else "#111827")
    return ax


def heatmap(matrix, xlabels=None, ylabels=None, ax=None, title: str | None = None,
            cmap: str = "magma", annotate: bool = False, fmt: str = "{:.2f}"):
    """A generic labelled heatmap — bigram counts, attention weights, and so on."""
    ax = ax or plt.gca()
    m = np.asarray(matrix, dtype=float)
    im = ax.imshow(m, cmap=cmap, aspect="auto")
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.grid(False)
    if title:
        ax.set_title(title)
    if annotate:
        big = m.max() if m.max() else 1
        for (r, c), v in np.ndenumerate(m):
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=6,
                    color="white" if v > big * 0.5 else "#111827")
    return im
