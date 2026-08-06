"""Checks that the *teaching* works, not only that the code runs.

The page tests prove a chapter doesn't crash. They cannot tell you that a chapter is
teaching something impossible — which is exactly what happened here: chapter 00 dealt six
random creatures, all of which happened to be "not a zeep", leaving a reader with no
positive example and a rule that could not be worked out. It ran perfectly. It was also
the first five minutes of the course.

Anything a reader is asked to deduce gets a test here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from kidsml.zeeps import (
    RULES,
    all_zeeps,
    label_with,
    quiz_examples,
    teaching_examples,
)

SEEDS = range(8)


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("seed", SEEDS)
def test_examples_show_both_answers(rule: str, seed: int):
    """A pile of examples that are all 'no' teaches nothing at all."""
    creatures = all_zeeps()
    labels = label_with(creatures, rule)
    shown = teaching_examples(rule, n=6, seed=seed)

    assert labels[shown].sum() >= 2, (
        f"rule {rule!r}, seed {seed}: fewer than two zeeps among the examples — "
        "there is nothing positive to generalise from"
    )
    assert (~labels[shown]).sum() >= 2, (
        f"rule {rule!r}, seed {seed}: fewer than two non-zeeps among the examples"
    )


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("seed", SEEDS)
def test_examples_are_distinct(rule: str, seed: int):
    shown = teaching_examples(rule, n=6, seed=seed)
    assert len(set(shown.tolist())) == len(shown), "the same creature shown twice"


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_quiz_asks_about_unseen_creatures(rule: str, seed: int):
    """Quizzing on a creature they were already given the answer for proves nothing."""
    shown = teaching_examples(rule, n=6, seed=seed)
    quiz = quiz_examples(rule, shown, n=3, seed=seed)

    assert not set(quiz.tolist()) & set(shown.tolist()), "the quiz reuses a shown example"
    assert len(set(quiz.tolist())) == len(quiz), "the same creature quizzed twice"


@pytest.mark.parametrize("rule", list(RULES))
def test_a_near_miss_pair_is_present_when_one_exists(rule: str):
    """The pair that differs in one way and still disagrees is what kills wrong guesses."""
    creatures = all_zeeps()
    labels = label_with(creatures, rule)
    rows = [tuple(r) for r in creatures.to_numpy()]

    exists = any(
        sum(1 for a, b in zip(rows[i], rows[j]) if a != b) == 1
        for i in range(len(rows))
        for j in range(len(rows))
        if labels[i] and not labels[j]
    )
    if not exists:
        pytest.skip(f"rule {rule!r} has no near-miss pair to include")

    shown = teaching_examples(rule, n=6, seed=0)
    found = any(
        sum(1 for a, b in zip(rows[i], rows[j]) if a != b) == 1
        for i in shown
        for j in shown
        if labels[i] and not labels[j]
    )
    assert found, f"rule {rule!r}: the examples never pin the rule down"


def test_the_vowels_really_do_cluster():
    """Chapter 23's payoff: nobody tells the model what a vowel is, and it groups them.

    This is the most striking claim in the course, so it gets checked rather than hoped
    for. If a change to the model or the training makes the letters scatter, the chapter
    stops being true and this test says so.
    """
    import numpy as np

    from kidsml.datasets import load_words
    from kidsml.langmodels import embedding_points, train_mlp_language_model
    from kidsml.text import VOWELS

    bundle = train_mlp_language_model(
        load_words("names"), block_size=3, embed_dim=2, hidden=96,
        n_words=8000, steps=1200, lr=0.03, seed=1,
    )

    points = np.asarray(embedding_points(bundle), dtype=float)
    points = (points - points.mean(axis=0)) / (points.std(axis=0) + 1e-9)
    is_vowel = np.array([c in VOWELS for c in bundle.vocab.chars])

    vowels = np.where(is_vowel)[0]
    others = np.where(~is_vowel)[0]

    among_vowels = np.mean([
        np.linalg.norm(points[i] - points[j]) for i in vowels for j in vowels if i < j
    ])
    vowel_to_other = np.mean([
        np.linalg.norm(points[i] - points[j]) for i in vowels for j in others
    ])

    assert among_vowels < vowel_to_other * 0.85, (
        f"the vowels stopped clustering: {among_vowels:.2f} apart from each other versus "
        f"{vowel_to_other:.2f} from everything else — chapter 23's best moment is gone"
    )


# ---------------------------------------------------------------------------
# Claims the chapters make out loud, checked against what the code actually does
# ---------------------------------------------------------------------------


def test_the_creature_tree_asks_about_wings_first():
    """Chapter 05's best moment: the reader picks a first question, and sklearn agrees.

    If sklearn ever picked a different column, the chapter would be telling the reader
    they matched the machine when they hadn't.
    """
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    from kidsml.datasets import load_table

    creatures = load_table("creatures")
    features = pd.get_dummies(creatures.drop(columns=["name", "can_fly"]))
    answers = (creatures["can_fly"] == "yes").astype(int)

    tree = DecisionTreeClassifier(random_state=0).fit(features, answers)
    first_question = features.columns[tree.tree_.feature[0]]

    assert "wings" in first_question, (
        f"chapter 05 says the machine asks about wings first; it asked about "
        f"{first_question!r}"
    )


def test_the_monster_model_finds_the_secret_rule():
    """Chapter 06 and 10 reveal the rule and invite the reader to check the model found it."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    from kidsml.datasets import load_table

    monsters = load_table("monsters")
    features = pd.get_dummies(monsters.drop(columns=["name", "is_boss"]))
    answers = (monsters["is_boss"] == "yes").astype(int)

    forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(features, answers)
    ranked = pd.Series(forest.feature_importances_, index=features.columns).sort_values(
        ascending=False
    )

    assert set(ranked.head(3).index) == {"attack", "magic", "speed"}, (
        f"the rule uses attack, magic and speed, but the model leaned on "
        f"{list(ranked.head(3).index)} — the reveal would not land"
    )


def test_the_ten_by_hand_points_can_actually_be_separated():
    """Chapter 02 asks the reader to split these ten points with one line, by hand."""
    from sklearn.linear_model import Perceptron

    from kidsml.datasets import two_blobs_tiny

    points, labels = two_blobs_tiny()
    model = Perceptron(max_iter=2000, tol=None).fit(points, labels)

    assert model.score(points, labels) == 1.0, (
        "chapter 02 asks for a perfect line on these points, so one has to exist"
    )


def test_q_is_followed_by_u():
    """Chapter 22's first exploration of the tally table, and its first small delight."""
    import numpy as np

    from kidsml.datasets import load_words
    from kidsml.text import CharVocab, bigram_counts, counts_to_probs

    words = load_words("names")
    vocab = CharVocab.from_words(words)
    probs = counts_to_probs(bigram_counts(words, vocab))

    after_q = probs[vocab.stoi["q"]]
    likeliest = vocab.itos[int(np.argmax(after_q))]

    assert likeliest == "u", f"chapter 22 promises q leads to u; the data says {likeliest!r}"


# ---------------------------------------------------------------------------
# Samples shown to the reader must show more than one answer
# ---------------------------------------------------------------------------
#
# A table of five examples that are all the same class looks fine, runs fine, and teaches
# the reader nothing — there is no contrast to notice. This bit us three times: chapter 00
# dealt six all-negative creatures, chapters 02 and 13 took the first five dogs (all
# puppies), and chapter 10 opened the penguin data with eight Adelie in a row.


def test_the_by_hand_dogs_include_both_answers():
    """Chapters 02 and 13 walk through five of the ten dogs by hand."""
    from kidsml.datasets import two_blobs_tiny

    _, labels = two_blobs_tiny()
    rows = [0, 3, 5, 7, 9]  # the selection both chapters use

    shown = labels[rows]
    assert set(shown.tolist()) == {0, 1}, (
        "the by-hand dogs are all the same kind, so the score never changes sign and "
        "there is nothing for the reader to notice"
    )


def test_the_first_look_at_the_penguins_shows_every_species():
    """Chapter 10 asks the reader to tell three species apart, so all three must appear."""
    from kidsml.datasets import load_table

    penguins = load_table("penguins")
    sample = penguins.groupby("species", group_keys=False).head(3)

    assert sample["species"].nunique() == penguins["species"].nunique(), (
        "the opening sample of the penguin data misses a species — the file is sorted, "
        "so taking the first rows shows only one"
    )


def test_plotly_defaults_to_the_dark_template():
    """Plotly draws on white, which is a rectangle of glare in this app.

    Checked at the source — the default template — rather than at each call site, because
    most figures are built inside helper functions and a call-site check missed them.
    """
    import plotly.io as pio

    from kidsml.plots import BACKGROUND, PANEL, use_house_style

    use_house_style()

    assert "kidsml" in pio.templates.default, (
        f"plotly's default template is {pio.templates.default!r}; figures will render white"
    )

    resolved = pio.templates[pio.templates.default].layout
    assert resolved.paper_bgcolor == BACKGROUND
    assert resolved.plot_bgcolor == PANEL


def test_plotly_renderer_cannot_open_a_browser():
    """fig.show() sniffing the environment once turned a 3s notebook into a 200s hang."""
    import plotly.io as pio

    from kidsml.plots import use_house_style

    use_house_style()
    assert pio.renderers.default == "plotly_mimetype", (
        f"renderer is {pio.renderers.default!r}; anything that can fall back to a browser "
        "will block under headless execution"
    )


# ---------------------------------------------------------------------------
# No term reaches the reader before the course explains it
# ---------------------------------------------------------------------------

COURSE_TERMS = [
    "gradient", "loss", "perceptron", "sigmoid", "probability", "impurity", "gini",
    "stump", "residual", "ensemble", "margin", "support vector", "kernel", "gamma",
    "baseline", "cross-validation", "fold", "one-hot", "precision", "recall",
    "confusion matrix", "leakage", "determinant", "projection", "neuron", "activation",
    "backpropagation", "chain rule", "learning rate", "epoch", "hidden layer",
    "overfitting", "weight decay", "early stopping", "tensor", "autograd", "optimizer",
    "pixel", "channel", "convolution", "feature map", "pooling", "centroid", "inertia",
    "principal component", "t-sne", "corpus", "vocabulary", "bigram", "smoothing",
    "context window", "embedding", "softmax", "temperature", "attention", "causal mask",
    "transformer", "hallucination",
]

_EXPLAINS = re.compile(
    r"(means|is when|is a |is the |called|think of|in other words|grown-ups call|"
    r"name for|stands for|which is|that is|= )",
    re.I,
)


@pytest.mark.parametrize("term", COURSE_TERMS, ids=lambda t: t.replace(" ", "_"))
def test_term_is_explained_where_the_reader_first_meets_it(term: str):
    """A word the reader has never seen must be explained where it first appears.

    "First" means first in the whole course, not first in the chapter — the reader only
    gets one first time. This is a rough check (it looks for explanatory phrasing near the
    first mention), so it can miss a bad explanation. It cannot miss a *missing* one.
    """
    import pathlib

    pages = sorted((ROOT / "app" / "pages").glob("[0-9][0-9]_*.py"))
    for page in pages:
        source = page.read_text(encoding="utf-8")
        prose = " ".join(
            re.findall(
                r'lesson\.(?:say|look_for|jargon|aha|careful|kid_corner)\(\s*\n?\s*"{1,3}(.*?)"{1,3}',
                source,
                re.S,
            )
        )
        named = " ".join(re.findall(r'lesson\.jargon\(\s*\n?\s*"([^"]+)"', source))

        match = re.search(rf"\b{re.escape(term)}", prose, re.I)
        if not match:
            continue

        nearby = prose.lower()[max(0, match.start() - 260) : match.start() + 260]
        explained = bool(_EXPLAINS.search(nearby)) or term.lower() in named.lower()
        assert explained, (
            f"{term!r} first reaches the reader in {page.name} with nothing nearby that "
            "explains it. The reader only gets one first time."
        )
        return
