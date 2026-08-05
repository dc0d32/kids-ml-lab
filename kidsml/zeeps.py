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


def _differs_in_one_way(a, b) -> bool:
    """True when two creatures are identical except for a single attribute."""
    return sum(1 for x, y in zip(a, b) if x != y) == 1


def teaching_examples(rule: str, n: int = 6, seed: int = 0):
    """Pick examples the rule can actually be worked out from.

    Drawing at random is a trap. Some rules are true of only 3 creatures out of 18, so a
    random handful can easily come back all-negative — and then there is nothing to spot,
    only something to be confused by. That is a bad first five minutes for a nervous
    reader, and it is exactly what used to happen here.

    So we deal on purpose:

    1. Start with a **near-miss pair** — two creatures that differ in one single way and
       still get different answers. That pair is what kills every wrong guess, and it is
       the thing an expert would look for first.
    2. Fill the rest keeping both answers on the table.

    Returns the row positions to show.
    """
    creatures = all_zeeps()
    labels = label_with(creatures, rule)
    rows = [tuple(row) for row in creatures.to_numpy()]
    rng = np.random.default_rng(seed)

    yes = [i for i in range(len(rows)) if labels[i]]
    no = [i for i in range(len(rows)) if not labels[i]]

    chosen: list[int] = []
    pairs = [(i, j) for i in yes for j in no if _differs_in_one_way(rows[i], rows[j])]
    if pairs:
        chosen = list(pairs[int(rng.integers(len(pairs)))])

    # At least a third of the examples should be "yes", so the reader has something
    # positive to generalise from rather than only a list of things it isn't.
    wanted_yes = max(2, n // 3)

    def top_up(pool, how_many):
        spare = [i for i in pool if i not in chosen]
        rng.shuffle(spare)
        return spare[:how_many]

    chosen += top_up(yes, wanted_yes - sum(1 for i in chosen if labels[i]))
    chosen += top_up(no, n - len(chosen))

    chosen = chosen[:n]
    rng.shuffle(chosen)
    return np.array(chosen, dtype=int)


def quiz_examples(rule: str, shown, n: int = 3, seed: int = 0):
    """Creatures held back for the quiz, with both answers represented where possible."""
    creatures = all_zeeps()
    labels = label_with(creatures, rule)
    rng = np.random.default_rng(seed + 991)

    left = [i for i in range(len(creatures)) if i not in set(int(s) for s in shown)]
    yes = [i for i in left if labels[i]]
    no = [i for i in left if not labels[i]]
    rng.shuffle(yes)
    rng.shuffle(no)

    picked = yes[:1] + no[:1]
    rest = [i for i in left if i not in picked]
    rng.shuffle(rest)
    picked += rest[: n - len(picked)]
    picked = picked[:n]
    rng.shuffle(picked)
    return np.array(picked, dtype=int)


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
