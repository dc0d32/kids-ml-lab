"""Chapter 05 · Twenty Questions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from kidsml import lesson, ui
from kidsml.datasets import load_table
from kidsml.plots import decision_boundary
from kidsml.trees import (
    creature_feature_names,
    creature_split_table,
    fit_creature_tree,
    fit_tree_shape,
    mushroom_tree,
    plot_decision_tree,
    shallow_mushroom_scores,
    tree_depth_scores,
)

lesson.begin(5)


@st.cache_data(show_spinner=False)
def cached_creature_splits():
    return creature_split_table()


@st.cache_data(show_spinner=False)
def cached_depth_scores(shape, n, noise, seed):
    return tree_depth_scores(shape=shape, n=n, noise=noise, seed=seed)


@st.cache_data(show_spinner=False)
def cached_mushroom_scores():
    return shallow_mushroom_scores()


@lesson.step("You already know this game", beat="hook")
def _():
    lesson.say(
        """
Time for a model with a clipboard. In Chapter 3, a straight line failed on circles and XOR. We had two escapes:
**invent new features**, or use a **bendy model**. This chapter is the first
bendy model: it bends by asking questions.
"""
    )
    lesson.mermaid(
        """
graph TD
    A[Start with one pile] --> B{Has wings?}
    B -->|yes| C{Has feathers?}
    B -->|no| D[mostly cannot fly]
    C -->|yes| E{Lives in water?}
    C -->|no| F[can fly]
""",
        height=270,
    )
    lesson.look_for("one row walking one path. It never answers every question in the diagram.")
    lesson.jargon("decision tree", "A model that asks yes/no questions until it reaches an answer.")


@lesson.step("Ten creatures. Four first questions", beat="byhand")
def _():
    lesson.say(
        """
Here are ten made-up creatures. We want to guess `can_fly`, but the tree is only
allowed to start with **one** column.
"""
    )
    creatures = load_table("creatures")
    st.dataframe(creatures, hide_index=True, width="stretch")
    lesson.look_for("columns that might split flyers away from non-flyers in one question.")


@lesson.step("Pick the least-mixed split", beat="byhand")
def _():
    lesson.say(
        """
A bucket is **mixed** when different answers are still stuck together. Six
animals with 3 flyers and 3 non-flyers is very mixed. Four flyers and 0
non-flyers is clean.
"""
    )
    splits = cached_creature_splits()
    st.dataframe(splits, hide_index=True, width="stretch")
    lesson.look_for("the lowest weighted mix. That is the question leaving the least mess for later.")
    lesson.say(
        """
For `has_wings`, the yes bucket has 6 animals and mix `0.444`, while the no
bucket has 4 animals and mix `0`. So the split score is `(6×0.444 + 4×0) / 10 = 0.267`.
"""
    )


@lesson.step("Will sklearn pick your question?", beat="seeit")
def _():
    splits = cached_creature_splits()
    winner = splits.iloc[0]["first question"]
    choices = creature_feature_names()
    guess = lesson.predict(
        "Which first question will sklearn put at the top of its tree?",
        choices,
        correct=choices.index(winner),
        why="It builds the split table like a sorting tray: pour rows into buckets, measure the leftover mix, then choose the question with the smallest mess.",
        key="ch05_first_split",
    )
    if guess is None:
        return

    model, _, _ = fit_creature_tree(max_depth=3)
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_decision_tree(model, creature_feature_names(), ["cannot fly", "can fly"], ax=ax)
    lesson.show(fig)
    lesson.look_for("the top box. It matches the least-mixed first question from the table.")
    first = creature_feature_names()[int(model.tree_.feature[0])]
    st.info(f"The computer's first split is **{first}** too. The split table did, as the adults apparently say, let him cook.")


@lesson.step("A tree bends by making stairs", beat="play")
def _():
    lesson.say(
        """
On a two-column picture, a question such as `x1 <= 0.4` makes a vertical cut. A
question such as `x2 <= -0.2` makes a horizontal cut.
"""
    )
    shape = ui.shape_picker(default="moons", key="ch05_shape", include=("moons", "circles", "xor", "spiral"))
    noise = ui.noise_slider(default=0.20, key="ch05_noise")
    n = ui.sample_slider(default=220, key="ch05_n")
    seed = ui.seed_slider(default=1, key="ch05_seed")
    depth = st.slider("How many questions deep?", 1, 20, 3, key="ch05_depth")
    model, X_train, X_test, y_train, y_test = fit_tree_shape(shape, depth, n=n, noise=noise, seed=seed)
    fig, ax = lesson.figure(6.5, 5.2)
    decision_boundary(model.predict, X_train, y_train, ax=ax, steps=180, shade_confidence=False, title=f"max_depth = {depth}")
    lesson.show(fig)
    lesson.look_for("the boundary made only of horizontal and vertical cuts. More depth adds more little rectangles.")
    col1, col2 = st.columns(2)
    col1.metric("training accuracy", f"{model.score(X_train, y_train):.0%}")
    col2.metric("test accuracy", f"{model.score(X_test, y_test):.0%}")


@lesson.step("Train and test split apart", beat="play")
def _():
    guess = lesson.predict(
        "As a tree gets deeper and deeper, what usually happens to training score and test score?",
        ["Both rise forever", "Training can keep rising while test levels off or falls", "Both fall together"],
        correct=1,
        why="A deep tree can fence off tiny boxes around training dots, even oddball dots. Training score climbs; next-data score can skid because the boxes copied noise.",
        key="ch05_depth_gap",
    )
    if guess is None:
        return

    shape = ui.shape_picker(default="moons", key="ch05_curve_shape", include=("moons", "circles", "xor", "spiral"))
    noise = ui.noise_slider(default=0.20, key="ch05_curve_noise")
    n = ui.sample_slider(default=220, key="ch05_curve_n")
    seed = ui.seed_slider(default=1, key="ch05_curve_seed")
    scores = cached_depth_scores(shape, n, noise, seed).set_index("max_depth")
    st.line_chart(scores)
    lesson.look_for("the gap between training accuracy and test accuracy. The gap is the over-studying warning.")
    lesson.aha("A tree is **blocky-bendy**: small straight cuts can wrap around moons, circles, or XOR without inventing new features.")


@lesson.step("Words become yes/no columns", beat="forreal")
def _():
    lesson.say(
        """
Real tables often contain words. A column that says `smell = almond` cannot go
straight into a tree as a sentence, so we turn it into yes/no columns like
`smell_almond`, `smell_fishy`, and `smell_none`.
"""
    )
    lesson.jargon("one-hot encoding", "Turning one word column into many yes/no columns the tree can ask about.")


@lesson.step("Mushroom questions", beat="forreal")
def _():
    mush_depth = st.slider("Mushroom tree depth", 1, 8, 4, key="ch05_mush_depth")
    _, _, _, _, _, scores, text = mushroom_tree(max_depth=mush_depth)
    cols = st.columns(3)
    cols[0].metric("train", f"{scores['train']:.1%}")
    cols[1].metric("test", f"{scores['test']:.1%}")
    cols[2].metric("top question", scores["top_question"].replace("_", " "))
    st.dataframe(cached_mushroom_scores(), hide_index=True, width="stretch")
    st.text(text[:2200])
    lesson.look_for("the top question before you read the text tree. Smell sits near the top, and real mushroom guides talk about smell too.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. Find the shallowest mushroom tree that stays above **95%** on the test set.
2. Find the tree depth where training and test accuracy are farthest apart.
3. Turn one mushroom rule into a sentence you could tell a person.
"""
    )
    lesson.kid_corner("Play Guess Who with animals. What is the best first question?")


lesson.finish()
