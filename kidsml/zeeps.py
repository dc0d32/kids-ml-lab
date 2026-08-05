"""The zeep game from Chapter 00.

A tiny made-up world: every creature has a shape, a colour and a size. A *secret rule*
decides whether it counts as a "zeep". There are only 18 possible creatures in the whole
universe, which is the point — you can hold the entire dataset in your head.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

SHAPES = ("circle", "square", "triangle")
COLOURS = ("red", "blue", "green")
SIZES = ("small", "big")

# Each rule is a plain English description plus a function over one creature.
RULES: dict[str, str] = {
    "is_red": "it is red",
    "big_square": "it is big AND a square",
    "not_green": "it is NOT green",
    "triangle_or_small": "it is a triangle OR it is small",
    "red_xor_big": "exactly one of these is true: it is red / it is big",
}

_RULE_FUNCS = {
    "is_red": lambda r: r.colour == "red",
    "big_square": lambda r: (r.size_ == "big") and (r.shape == "square"),
    "not_green": lambda r: r.colour != "green",
    "triangle_or_small": lambda r: (r.shape == "triangle") or (r.size_ == "small"),
    "red_xor_big": lambda r: (r.colour == "red") != (r.size_ == "big"),
}


def all_zeeps() -> pd.DataFrame:
    """Every creature that can exist in this world. All 18 of them."""
    rows = [
        {"shape": s, "colour": c, "size_": z}
        for s in SHAPES
        for c in COLOURS
        for z in SIZES
    ]
    return pd.DataFrame(rows)


def pretty(creatures: pd.DataFrame) -> pd.DataFrame:
    """The same table with a nicer column name for showing on screen.

    (The column is called ``size_`` in code because ``size`` is already the name of a
    pandas attribute, and that clash causes confusing bugs.)
    """
    return creatures.rename(columns={"size_": "size"})


def label_with(creatures: pd.DataFrame, rule: str) -> np.ndarray:
    """Apply a secret rule to every creature. Returns an array of True/False."""
    func = _RULE_FUNCS[rule]
    return np.array([func(row) for row in creatures.itertuples()], dtype=bool)


def encode(creatures: pd.DataFrame) -> np.ndarray:
    """Words to numbers, because a model can only do arithmetic.

    ``circle`` becomes 0, ``square`` becomes 1, and so on. The model has no idea that
    0 means circle — and it doesn't need to.
    """
    lookup = [
        {v: i for i, v in enumerate(SHAPES)},
        {v: i for i, v in enumerate(COLOURS)},
        {v: i for i, v in enumerate(SIZES)},
    ]
    cols = ["shape", "colour", "size_"]
    return np.array(
        [[lookup[j][row[c]] for j, c in enumerate(cols)] for _, row in creatures.iterrows()]
    )


def learning_curve(rule: str, n_repeats: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    """How often is the computer right, given N examples to learn from?

    For every possible number of examples we deal the cards many times over and take
    the average, so one lucky deal doesn't fool us.
    """
    creatures = all_zeeps()
    X = encode(creatures)
    y = label_with(creatures, rule).astype(int)
    rng = np.random.default_rng(seed)

    ns, accs = [], []
    for n in range(1, len(creatures)):
        scores = []
        for _ in range(n_repeats):
            order = rng.permutation(len(creatures))
            train, test = order[:n], order[n:]
            if len(np.unique(y[train])) < 2:
                # Only one kind of answer in the examples: the model can only ever say
                # that one thing. That is a real result, not a bug — keep it.
                scores.append(float((y[test] == y[train][0]).mean()))
                continue
            model = DecisionTreeClassifier(random_state=0).fit(X[train], y[train])
            scores.append(float((model.predict(X[test]) == y[test]).mean()))
        ns.append(n)
        accs.append(float(np.mean(scores)))

    return {"n": np.array(ns), "accuracy": np.array(accs)}
