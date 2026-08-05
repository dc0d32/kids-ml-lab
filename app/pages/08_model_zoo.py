"""Chapter 08 · The Model Zoo."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

from kidsml import ui
from kidsml.trees import (
    MODEL_PERSONALITIES,
    deep_tree_train_test,
    fold_scores,
    lopsided_baseline,
    penguin_leaderboard,
    plot_folds,
    plot_zoo,
    split_bounce_scores,
)

ui.page_setup(8)


@st.cache_data(show_spinner=False)
def cached_zoo(shape, n, noise, seed):
    return plot_zoo(shape=shape, n=n, noise=noise, seed=seed)


@st.cache_data(show_spinner=False)
def cached_leaderboard():
    return penguin_leaderboard()


# ---------------------------------------------------------------------------
ui.beat("hook", "Which guesser should you use?")

st.markdown(
    """
You now know several guessers: lines, probabilities, trees, crowds, and widest roads.
Which one should you use?

The honest answer is: **try them and see**.

But *see* is harder than it sounds. This chapter is mostly about not lying to yourself.
"""
)

ui.little_kid_corner(
    "If you test a bike, a scooter, and skates, use the same hill for all three. A fair race needs fair rules."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Five practice tests.")

st.markdown(
    "Cut 10 rows into 5 folds of 2 rows. Each round hides one fold as the test set."
)
fold_table = pd.DataFrame(
    {
        "round": [1, 2, 3, 4, 5],
        "test rows": ["1, 2", "3, 4", "5, 6", "7, 8", "9, 10"],
        "score": [0.80, 0.70, 0.90, 0.80, 0.60],
    }
)
st.dataframe(fold_table, hide_index=True, use_container_width=True)
st.info("Average score = (0.80 + 0.70 + 0.90 + 0.80 + 0.60) / 5 = 0.76")

ui.jargon("cross-validation", "Take turns hiding different chunks, then report the average and spread.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "The zoo has no champion for every shape.")

shape = ui.shape_picker(default="moons", key="ch08_shape")
noise = ui.noise_slider(default=0.20, key="ch08_noise")
n = ui.sample_slider(default=180, key="ch08_n")
seed = ui.seed_slider(default=0, key="ch08_seed")
fig = cached_zoo(shape, n, noise, seed)
ui.show(fig)

st.markdown("**Model personalities:**")
st.dataframe(pd.DataFrame({"model": list(MODEL_PERSONALITIES), "personality": list(MODEL_PERSONALITIES.values())}), hide_index=True)
ui.aha("No model wins on every shape. That is not a cop-out. It is the real state of the field.")

# ---------------------------------------------------------------------------
ui.beat("play", "How to not fool yourself.")

st.markdown("If I let you study the exact test questions, your score means nothing.")
st.dataframe(pd.DataFrame([deep_tree_train_test()]), hide_index=True, use_container_width=True)

ui.careful(
    "Evaluating on the training data is a fake victory. A deep tree can score 100% there "
    "and still miss new points."
)

test_size = st.slider("How much data goes in the test set?", 0.15, 0.50, 0.30, 0.05, key="test_size")
bounce = split_bounce_scores(test_size=test_size, max_seed=10).set_index("seed")
st.line_chart(bounce)
st.caption("The same model bounces because different rows land in the test set.")

scores = fold_scores()
fig, ax = ui.figure(7, 3.6)
plot_folds(ax=ax)
ui.show(fig)
st.metric("5-fold average", f"{scores.mean():.1%}", f"spread ±{scores.std():.1%}")
st.write([f"{s:.1%}" for s in scores])

baseline = lopsided_baseline()
st.warning(f"A useless 'always say the most common class' baseline scores {baseline:.0%} on a lopsided dataset. Chapter 10 digs into that trap.")

# ---------------------------------------------------------------------------
ui.beat("forreal", "An honest penguin leaderboard.")

leaderboard = cached_leaderboard()
st.dataframe(leaderboard, hide_index=True, use_container_width=True)
st.markdown(
    "If two means are closer than their spreads, do not brag that one crushed the other. "
    "A score without a spread is half a fact."
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find a shape where logistic regression wins or ties.
2. Find a seed where a bad model looks good.
3. Make two models swap places by changing only the split seed.
4. 🧸 **Little Kid Corner:** Race three toys down the same ramp three times. Report the average and the wiggle.
"""
)

ui.worksheet_link(8)
