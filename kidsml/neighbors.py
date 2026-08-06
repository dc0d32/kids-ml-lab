"""Chapter 08 helpers · k nearest neighbours.

These live here (not in ``unsupervised.py``) because Chapter 08 moved into Part 1 and its
accuracy curve needed rebuilding so the reader can actually *see* the sweet spot: k = 1
memorises the noise, a middling k smooths it away, and a huge k collapses the whole plane
into one sleepy answer.

Everything the boundary pictures and the accuracy curve show comes from the *same* toy
dataset, so a reader can connect "the boundary at this k" with "the score at this k".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from kidsml.datasets import toy_shape
from kidsml.plots import ACCENT, COOL, WARM, decision_boundary

# One dataset for the whole "morph the boundary" story. Noisy moons make all three
# failure modes visible: islands at k = 1, a clean smooth boundary in the middle, and a
# collapse into one colour at huge k.
CURVE_SHAPE = "moons"
CURVE_NOISE = 0.45
CURVE_N = 240
CURVE_SEEDS = (7, 11, 23)

# k values swept for the curve: dense where the action is, then climbing to a large
# fraction of the training set so the collapse at the far end is on screen.
CURVE_KS = (1, 2, 3, 5, 7, 10, 15, 21, 31, 45, 65, 91, 125, 165)

# k values the reader can flip between in the app.
BOUNDARY_KS = (1, 7, 21, 121)


def knn_boundary_data(seed: int = CURVE_SEEDS[0]):
    """The single noisy-moons dataset used for both the boundary and the curve."""
    return toy_shape(CURVE_SHAPE, n=CURVE_N, noise=CURVE_NOISE, seed=seed)


def knn_accuracy_curve() -> pd.DataFrame:
    """Held-out accuracy for a sweep of k, averaged over folds and several datasets.

    Each k is scored with 5-fold cross-validation (so every point is judged on data it
    did not see) and then averaged across a few random datasets, which sands the wobble
    off the line so the hump is unmistakable.
    """
    totals = np.zeros(len(CURVE_KS))
    for seed in CURVE_SEEDS:
        X, y = toy_shape(CURVE_SHAPE, n=CURVE_N, noise=CURVE_NOISE, seed=seed)
        for i, k in enumerate(CURVE_KS):
            totals[i] += cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean()
    accuracy = totals / len(CURVE_SEEDS)
    return pd.DataFrame({"k": list(CURVE_KS), "test accuracy": accuracy})


def best_k(curve: pd.DataFrame | None = None) -> int:
    curve = knn_accuracy_curve() if curve is None else curve
    return int(curve.loc[curve["test accuracy"].idxmax(), "k"])


def plot_knn_accuracy_curve(curve: pd.DataFrame | None = None):
    """House-style figure of the accuracy hump, with the best k and both cliffs marked."""
    curve = knn_accuracy_curve() if curve is None else curve
    ks = curve["k"].to_numpy()
    acc = curve["test accuracy"].to_numpy()
    top = int(np.argmax(acc))
    best = int(ks[top])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(ks, acc, marker="o", color=COOL, linewidth=2.2, markersize=5, zorder=3)

    ax.axvline(best, color=ACCENT, linestyle="--", linewidth=1.8, alpha=0.9, zorder=2)
    ax.scatter([best], [acc[top]], s=150, color=ACCENT, edgecolors="black", linewidths=0.8, zorder=4)
    ax.annotate(
        f"best k = {best}",
        xy=(best, acc[top]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color=ACCENT,
        weight="bold",
        fontsize=11,
    )

    ax.annotate(
        "k = 1\nmemorises the noise",
        xy=(ks[0], acc[0]),
        xytext=(10, -46),
        textcoords="offset points",
        ha="left",
        color=WARM,
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color=WARM, lw=1.4),
    )
    ax.annotate(
        "huge k\none sleepy answer",
        xy=(ks[-1], acc[-1]),
        xytext=(-8, 30),
        textcoords="offset points",
        ha="right",
        color=WARM,
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color=WARM, lw=1.4),
    )

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100])
    ax.set_xticklabels(["1", "3", "10", "30", "100"])
    ax.set_xlabel("k  (how many neighbours vote)")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(min(acc) - 0.04, max(acc) + 0.06)
    ax.set_title("The sweet spot is in the middle")
    ax.grid(True, which="both", alpha=0.15)
    return fig


def plot_knn_boundary(k: int = 1, seed: int = CURVE_SEEDS[0]):
    """Decision boundary at a chosen k, on the same dataset the curve is scored on."""
    X, y = knn_boundary_data(seed)
    model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    decision_boundary(model.predict, X, y, ax=ax, steps=200, shade_confidence=False)
    ax.set_title(f"k = {k}")
    return fig
