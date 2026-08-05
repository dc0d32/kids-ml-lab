"""Chapter 06 · A Crowd of Trees."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.plots import decision_boundary
from kidsml.trees import (
    boosting_trace,
    fit_tree_and_forest,
    forest_vote_counts,
    monster_models,
    tiny_boosting_table,
    tiny_vote_table,
)

ui.page_setup(6)


@st.cache_data(show_spinner=False)
def cached_trace(steps, learning_rate, max_depth, seed):
    return boosting_trace(n_steps=steps, learning_rate=learning_rate, max_depth=max_depth, seed=seed)


@st.cache_data(show_spinner=False)
def cached_monsters():
    return monster_models()


# ---------------------------------------------------------------------------
ui.beat("hook", "A crowd can be wiser than one guesser.")

st.markdown(
    """
At a party, ask everyone to guess how many jellybeans are in a jar. One person may be
wildly wrong. The **average** of many guesses is often spooky-good.

Trees can do that too.

There are two crowd tricks: **vote**, or **take turns fixing mistakes**. Both are used
all over real machine learning.
"""
)

ui.little_kid_corner(
    "Ask five people where a hidden toy is. If four point under the couch, check there first. "
    "The crowd vote is stronger than one noisy guess."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Crowd trick 1: vote.")

st.markdown("Each tiny tree votes red or blue. You tally the majority.")
st.dataframe(tiny_vote_table(), hide_index=True, use_container_width=True)

st.markdown("Crowd trick 2: fix what is left over.")
st.dataframe(tiny_boosting_table(), hide_index=True, use_container_width=True)

ui.jargon("ensemble", "A model made by combining many smaller models.")
ui.jargon("residual", "The leftover mistake: actual answer minus current guess.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "One jagged tree beside a voting forest.")

shape = ui.shape_picker(default="moons", key="ch06_shape", include=("moons", "circles", "xor", "spiral"))
noise = ui.noise_slider(default=0.25, key="ch06_noise")
seed = ui.seed_slider(default=4, key="ch06_seed")
n_estimators = st.slider("How many trees vote?", 1, 80, 20, 1, key="forest_n")

X, y, tree, forest = fit_tree_and_forest(shape=shape, n_estimators=n_estimators, noise=noise, seed=seed)
fig, axes = ui.two_figures(4.8, 4.2)
decision_boundary(tree.predict, X, y, ax=axes[0], steps=150, shade_confidence=False, title="one tree")
decision_boundary(forest.predict, X, y, ax=axes[1], steps=150, shade_confidence=False, title=f"{n_estimators} trees voting")
ui.show(fig)

x1 = st.slider("Click-ish point: feature 1", -2.5, 2.5, 0.0, 0.1, key="vote_x1")
x2 = st.slider("Click-ish point: feature 2", -2.5, 2.5, 0.0, 0.1, key="vote_x2")
votes = forest_vote_counts(forest, [x1, x2])
st.info(f"For that point, **{votes['red']} of {n_estimators}** trees said red and **{votes['blue']}** said blue.")

# ---------------------------------------------------------------------------
ui.beat("play", "Boosting builds a curve from little steps.")

steps = st.slider("Boosting step", 1, 50, 12, key="boost_step")
rate = st.slider("How much of each fix to add", 0.05, 0.60, 0.25, 0.05, key="boost_rate")
stump_depth = st.slider("How deep is each tiny tree?", 1, 4, 1, key="boost_depth")
trace = cached_trace(50, rate, stump_depth, seed)
stage = trace["stages"][steps - 1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].scatter(trace["x"], trace["y"], color="#3B82F6", edgecolors="white", label="data")
axes[0].plot(trace["x_grid"], stage["running_grid"], color="#10B981")
axes[0].set_title("running prediction")
axes[1].scatter(trace["x"], stage["residual"], color="#EF4444", edgecolors="white")
axes[1].axhline(0, color="#94A3B8", linewidth=1.4)
axes[1].set_title("leftover mistakes")
axes[2].plot(trace["x_grid"], stage["newest_grid"], color="#10B981")
axes[2].set_title("newest little tree")
ui.show(fig)

ui.aha("A smooth-looking curve can be built out of many tiny step shapes.")
ui.careful("Boosting learns in order. Too many strong fixes can chase noise and wobble around the points.")

# ---------------------------------------------------------------------------
ui.beat("forreal", "Monsters have a secret rule.")

scores, importances, secret = cached_monsters()
st.dataframe(scores, hide_index=True, use_container_width=True)
st.bar_chart(importances.set_index("feature group"))

st.markdown(
    "Attack, magic, and speed rise to the top. Element, home, and height do not matter much. "
    "Now reveal the rule that made the data:"
)
st.code(secret)
st.warning("Five percent of the labels were flipped on purpose. A model scoring 100% here would be suspicious.")

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. How few trees does the forest need before it beats the one tree?
2. Make boosting overfit the wiggle: use many steps and deeper tiny trees. Where do the wobbles appear?
3. Find a monster feature the model ignores. Does the secret rule agree?
4. 🧸 **Little Kid Corner:** Guess jellybeans with a group. Compare one guess with the group average.
"""
)

ui.worksheet_link(6)
