"""Datasets for the course.

Two kinds live here:

1. **Toy shapes** — made-up 2D point clouds (blobs, moons, circles, XOR, spiral).
   They are tiny, instant, and you can *see* the whole dataset at once. Perfect for
   watching a decision boundary move.
2. **Real-ish tables** — small CSVs bundled in ``data/`` (penguins, mushrooms, Pokemon,
   bike rentals) so nothing has to be downloaded and no passwords are needed.

Every function returns plain NumPy arrays or a pandas DataFrame. Nothing clever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# 1D toy problems (for regression chapters)
# ---------------------------------------------------------------------------


def allowance(n: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Weeks of saving -> dollars in the piggy bank.

    A straight line with a little noise. Eight points, so they all fit on a
    hand-drawn graph and you can add up the errors with a pencil.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(1, n + 1, dtype=float)
    dollars = 3.0 * weeks + 5.0 + rng.normal(0, 2.0, size=n)
    return weeks, np.round(dollars, 1)


def wiggle(n: int = 40, noise: float = 0.25, seed: int = 0):
    """A curvy 1D dataset a straight line can never fit. Used to motivate boosting."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, size=n))
    y = np.sin(x * 1.4) + 0.35 * x + rng.normal(0, noise, size=n)
    return x, y


# ---------------------------------------------------------------------------
# 2D toy shapes (for classification chapters)
# ---------------------------------------------------------------------------

TOY_SHAPES = ("blobs", "moons", "circles", "xor", "spiral", "stripes")

SHAPE_BLURB = {
    "blobs": "Two clean clumps. A ruler handles this easily.",
    "moons": "Two crescents that hug each other. A ruler almost works.",
    "circles": "A ring around a dot. No straight line can ever split these.",
    "xor": "Four corners, opposite ones matching. The classic ruler-breaker.",
    "spiral": "Two arms wound together. Hard even for big models.",
    "stripes": "Alternating bands. Easy to describe, awkward to draw with one line.",
}


def toy_shape(
    name: str = "moons",
    n: int = 200,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for one of the toy 2D shapes.

    ``X`` has shape ``(n, 2)`` and is roughly centred on 0 with a spread of about 1,
    so the same plot limits work for every shape. ``y`` is 0 or 1.
    """
    if name not in TOY_SHAPES:
        raise ValueError(f"unknown shape {name!r}; pick one of {TOY_SHAPES}")

    rng = np.random.default_rng(seed)

    if name == "blobs":
        X, y = skds.make_blobs(
            n_samples=n, centers=[(-1.5, -1.0), (1.5, 1.0)], cluster_std=0.9 * (0.3 + noise), random_state=seed
        )
    elif name == "moons":
        X, y = skds.make_moons(n_samples=n, noise=noise * 0.5, random_state=seed)
    elif name == "circles":
        X, y = skds.make_circles(n_samples=n, noise=noise * 0.35, factor=0.45, random_state=seed)
    elif name == "xor":
        X = rng.uniform(-2, 2, size=(n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X = X + rng.normal(0, noise * 0.5, size=X.shape)
    elif name == "stripes":
        X = rng.uniform(-2, 2, size=(n, 2))
        y = (np.floor(X[:, 0] + 2) % 2).astype(int)
        X = X + rng.normal(0, noise * 0.35, size=X.shape)
    else:  # spiral
        half = n // 2
        t = np.sqrt(rng.uniform(0.06, 1.0, size=half)) * 2.6 * np.pi
        r = t / (2.6 * np.pi) * 2.0
        arm0 = np.c_[r * np.cos(t), r * np.sin(t)]
        arm1 = -arm0
        X = np.vstack([arm0, arm1]) + rng.normal(0, noise * 0.35, size=(2 * half, 2))
        y = np.r_[np.zeros(half, dtype=int), np.ones(half, dtype=int)]

    X = np.asarray(X, dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0).max()
    return X, np.asarray(y, dtype=int)


def xor_exact() -> tuple[np.ndarray, np.ndarray]:
    """The four XOR corners exactly — no noise, no extras.

    This is the dataset you can hold in your head: four points, four answers.
    """
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    return X, y


def two_blobs_tiny(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Ten points, two clumps, round numbers — small enough to classify by hand."""
    X = np.array(
        [
            [1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 2.0],
            [6.0, 5.0], [7.0, 6.0], [6.0, 7.0], [8.0, 6.0], [7.0, 8.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


# ---------------------------------------------------------------------------
# Clustering toys
# ---------------------------------------------------------------------------


def cluster_blobs(n: int = 240, k: int = 3, spread: float = 0.9, seed: int = 0):
    """Unlabelled-looking blobs for k-means. Returns ``(X, true_labels)``.

    The true labels exist only so a grown-up can check the answer — the chapter
    never shows them to the model.
    """
    X, y = skds.make_blobs(n_samples=n, centers=k, cluster_std=spread, random_state=seed)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def six_points_for_kmeans() -> np.ndarray:
    """Six whole-number points. One round of k-means on these is doable with a pencil."""
    return np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [7.0, 7.0], [8.0, 7.0], [7.0, 8.0]])


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def digits(n_classes: int = 10):
    """sklearn's 8x8 handwritten digits: 1797 images, 64 numbers each.

    Small enough that you can print one out as a grid of numbers and read it.
    Returns ``(X, y, images)`` where ``images`` is ``(n, 8, 8)``.
    """
    d = skds.load_digits(n_class=n_classes)
    return d.data.astype(float), d.target.astype(int), d.images.astype(float)


def tiny_image() -> np.ndarray:
    """A 5x5 image with an obvious vertical edge — for doing a convolution by hand."""
    return np.array(
        [
            [0, 0, 9, 9, 9],
            [0, 0, 9, 9, 9],
            [0, 0, 9, 9, 9],
            [0, 0, 9, 9, 9],
            [0, 0, 9, 9, 9],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Bundled real-ish tables
# ---------------------------------------------------------------------------

TABLES = {
    "penguins": ("penguins.csv", "species", "Guess the penguin species from its measurements."),
    "mushrooms": ("mushrooms.csv", "edible", "Edible or deadly? Get this one wrong and it matters."),
    "monsters": ("monsters.csv", "is_boss", "Is this trading-card monster a boss, judging by its stats?"),
    "bikes": ("bikes.csv", "rentals", "How many bikes get rented today, given the weather?"),
    "creatures": ("creatures.csv", "can_fly", "Ten made-up creatures. Small enough to build a tree by hand."),
}

# ``monsters`` is the one table where we know the true rule, because we invented it.
# Chapter 9 uses this to check whether a model actually found the pattern or just
# memorised the rows.
MONSTER_SECRET_RULE = "is_boss  ⟺  (attack + magic) > 150  AND  speed < 90   (plus 5% flipped labels)"


def load_table(name: str) -> pd.DataFrame:
    """Load one of the bundled CSVs from ``data/``."""
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}; pick one of {sorted(TABLES)}")
    return pd.read_csv(DATA_DIR / TABLES[name][0])


def target_of(name: str) -> str:
    """The column we are trying to predict for a bundled table."""
    return TABLES[name][1]


def blurb_of(name: str) -> str:
    """A one-line, kid-facing description of a bundled table."""
    return TABLES[name][2]


def features_and_target(name: str, features: list[str] | None = None):
    """Split a bundled table into ``(X_dataframe, y_series)``.

    Rows with missing values are dropped — the messy-data chapter talks about why.
    """
    df = load_table(name).dropna()
    target = target_of(name)
    if features is None:
        features = [c for c in df.columns if c != target]
    return df[features], df[target]


# ---------------------------------------------------------------------------
# Text corpora (for the generative chapters)
# ---------------------------------------------------------------------------

CORPORA = {
    "names": ("names.txt", "Ten thousand real first names, one per line."),
    "rhymes": ("rhymes.txt", "Traditional nursery rhymes and playground chants."),
    "fables": ("fables.txt", "Very short fables, each with a moral at the end."),
}


def load_corpus(name: str = "names") -> str:
    """Load one of the bundled text corpora as a single string."""
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; pick one of {sorted(CORPORA)}")
    return (DATA_DIR / CORPORA[name][0]).read_text(encoding="utf-8")


def load_words(name: str = "names") -> list[str]:
    """Load a corpus as a list of lowercase words/lines with the blanks removed."""
    text = load_corpus(name)
    return [w.strip().lower() for w in text.splitlines() if w.strip()]
