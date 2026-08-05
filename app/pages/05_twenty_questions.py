"""Chapter 05 · Twenty Questions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import ui
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

ui.page_setup(5)


@st.cache_data(show_spinner=False)
def cached_creature_splits():
    return creature_split_table()


@st.cache_data(show_spinner=False)
def cached_depth_scores(shape, n, noise, seed):
    return tree_depth_scores(shape=shape, n=n, noise=noise, seed=seed)


@st.cache_data(show_spinner=False)
def cached_mushroom_scores():
    return shallow_mushroom_scores()


# ---------------------------------------------------------------------------
ui.beat("hook", "You already know this game.")

st.markdown(
    """
In Chapter 3, a straight line failed on circles and XOR. We had two escapes:
**invent new features**, or use a **bendy model**.

This chapter is the first bendy model.

Think of **Twenty Questions** or **Guess Who**. Is it bigger than a cat? Does it have
wings? You split the pile, then ask another question. A decision tree is that game.
The clever part is choosing the first question.
"""
)

ui.little_kid_corner(
    "Put toy animals in a pile. Ask one yes/no question, like *does it have wings?* "
    "Move the yes toys left and the no toys right. Keep asking until each pile has one answer."
)

ui.jargon("decision tree", "A model that asks yes/no questions until it reaches an answer.")

# ---------------------------------------------------------------------------
ui.beat("byhand", "Ten creatures. Four possible first questions.")

creatures = load_table("creatures")
st.dataframe(creatures, hide_index=True, use_container_width=True)

st.markdown(
    """
A mixed bucket is messy. A clean bucket is good.

For two answers, the bucket mix score is:

`1 - p_yes² - p_no²`

A bucket with 3 flyers and 3 non-flyers has `1 - (3/6)² - (3/6)² = 0.5`.
A bucket with 4 flyers and 0 non-flyers has `1 - 1² - 0² = 0`.
"""
)

splits = cached_creature_splits()
st.dataframe(splits, hide_index=True, use_container_width=True)

winner = splits.iloc[0]["first question"]
st.success(f"The least-mixed first question is **{winner}**. Your pencil found it.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Now sklearn picks the same question.")

model, X_creatures, y_creatures = fit_creature_tree(max_depth=3)
fig, ax = plt.subplots(figsize=(11, 5))
plot_decision_tree(model, creature_feature_names(), ["cannot fly", "can fly"], ax=ax)
ui.show(fig)

st.code("""from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3)
model.fit(creature_questions, can_fly)""", language="python")

first = creature_feature_names()[int(model.tree_.feature[0])]
st.info(f"The computer's first split is **{first}** too.")

# ---------------------------------------------------------------------------
ui.beat("play", "A tree bends by making stairs.")

shape = ui.shape_picker(default="moons", key="ch05_shape", include=("moons", "circles", "xor", "spiral"))
noise = ui.noise_slider(default=0.20, key="ch05_noise")
n = ui.sample_slider(default=220, key="ch05_n")
seed = ui.seed_slider(default=1, key="ch05_seed")
depth = st.slider("How many questions deep?", 1, 20, 3, key="ch05_depth")

model, X_train, X_test, y_train, y_test = fit_tree_shape(shape, depth, n=n, noise=noise, seed=seed)
fig, ax = ui.figure(6.5, 5.2)
decision_boundary(model.predict, X_train, y_train, ax=ax, steps=180, shade_confidence=False, title=f"max_depth = {depth}")
ui.show(fig)

col1, col2 = st.columns(2)
col1.metric("training accuracy", f"{model.score(X_train, y_train):.0%}")
col2.metric("test accuracy", f"{model.score(X_test, y_test):.0%}")

scores = cached_depth_scores(shape, n, noise, seed).set_index("max_depth")
st.line_chart(scores)

ui.aha(
    "A tree is not smooth-bendy. It is **blocky-bendy**. Each question cuts only left-right "
    "or up-down. Enough little cuts can carve almost any shape."
)
ui.careful(
    "A depth-20 tree can get 100% on its own dots by memorising tiny boxes. That is like "
    "memorising last year's test answers. It feels great until the questions change."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "Mushrooms are a real table of words.")

st.markdown(
    "A column that says `smell` cannot be a number, so we make one yes/no column per smell. "
    "That is one-hot encoding."
)

mush_depth = st.slider("Mushroom tree depth", 1, 8, 4, key="mush_depth")
_, _, _, _, _, scores, text = mushroom_tree(max_depth=mush_depth)
cols = st.columns(3)
cols[0].metric("train", f"{scores['train']:.1%}")
cols[1].metric("test", f"{scores['test']:.1%}")
cols[2].metric("top question", scores["top_question"].replace("_", " "))

st.dataframe(cached_mushroom_scores(), hide_index=True, use_container_width=True)
st.text(text[:2200])

st.markdown(
    "Smell sits at the top. Real mushroom guides talk about smell too. The data and the "
    "foragers agree."
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find the shallowest mushroom tree that stays above **95%** on the test set.
2. Find the tree depth where training and test accuracy are farthest apart.
3. Turn one mushroom rule into a sentence you could tell a person.
4. 🧸 **Little Kid Corner:** Play Guess Who with animals. What is the best first question?
"""
)

ui.worksheet_link(5)
