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
You now know several guessers: lines, probabilities, trees, crowds, and widest
roads. Which one should you use?

The honest answer is: **try them and see**. But *see* is harder than it sounds,
because a model can look brilliant on the rows it studied and stumble on new
rows.

This chapter is about fair races. We compare models on data they did not train
on, we repeat the race, and we always ask what a boring baseline could score.
"""
)

ui.mermaid(
    """
graph LR
    A[all labelled rows] --> B[training rows]
    A --> C[hidden test rows]
    B --> D[train model]
    D --> E[predict hidden rows]
    C --> E
    E --> F[test score]
""",
    height=260,
)

st.markdown(
    """
Notice the wall between training rows and hidden test rows. If the model studies
the test rows, the score stops being evidence about new data.
"""
)

ui.little_kid_corner(
    "If you test a bike, a scooter, and skates, use the same hill for all three. "
    "A fair race needs fair rules."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Five practice tests.")

st.markdown(
    """
One train/test split can be lucky. Maybe the easy rows landed in the test set.
Maybe the hard rows did. Cross-validation turns that one race into several
smaller races.

Cut 10 rows into 5 folds of 2 rows. Each round hides one fold as the test set
and trains on the other four folds. Every row gets a turn being hidden.
"""
)

ui.mermaid(
    """
graph TD
    A[10 rows] --> B[5 folds]
    B --> C[round 1: fold 1 tests]
    B --> D[round 2: fold 2 tests]
    B --> E[more rounds rotate]
    C --> F[average and spread]
    D --> F
    E --> F
""",
    height=280,
)

st.markdown(
    """
The rotation is the point. A bouncy test score is not a bug; it is a warning
that one split is a shaky fact.
"""
)

fold_table = pd.DataFrame(
    {
        "round": [1, 2, 3, 4, 5],
        "test rows": ["1, 2", "3, 4", "5, 6", "7, 8", "9, 10"],
        "score": [0.80, 0.70, 0.90, 0.80, 0.60],
    }
)
st.dataframe(fold_table, hide_index=True, use_container_width=True)
st.info("Average score = (0.80 + 0.70 + 0.90 + 0.80 + 0.60) / 5 = 3.80 / 5 = 0.76")

ui.jargon("cross-validation", "Take turns hiding different chunks, then report the average and spread.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "The zoo has no champion for every shape.")

st.markdown(
    """
Here is the model zoo. Every model gets the same training rows and the same
hidden test rows. That keeps the race fair.

Watch the shapes. Lines like straight-ish borders. Trees like boxes. RBF SVMs
like smooth islands. kNN listens to nearby points. No personality wins every
kind of problem.
"""
)

shape = ui.shape_picker(default="moons", key="ch08_shape")
noise = ui.noise_slider(default=0.20, key="ch08_noise")
n = ui.sample_slider(default=180, key="ch08_n")
seed = ui.seed_slider(default=0, key="ch08_seed")
fig = cached_zoo(shape, n, noise, seed)
ui.show(fig)

st.markdown(
    """
Look at both parts of each mini-plot: the boundary shape and the test score in
the title. A model can have the wrong personality for one shape and the right
personality for another.
"""
)

st.markdown("**Model personalities:**")
st.dataframe(pd.DataFrame({"model": list(MODEL_PERSONALITIES), "personality": list(MODEL_PERSONALITIES.values())}), hide_index=True)
ui.aha("No model wins on every shape. That is not a cop-out. It is the real state of the field.")

# ---------------------------------------------------------------------------
ui.beat("play", "How to not fool yourself.")

st.markdown(
    """
Scoring a model on its own training data is like taking a practice test after
memorising the answer key. It may tell you the model stored the rows. It does
not tell you whether it learned a pattern that works on new rows.
"""
)
st.dataframe(pd.DataFrame([deep_tree_train_test()]), hide_index=True, use_container_width=True)

ui.careful(
    "Evaluating on the training data is a fake victory. A deep tree can score 100% there "
    "by memorising tiny boxes, then miss new points that do not land in those boxes."
)

st.markdown(
    """
Now change only the split seed. The model type and dataset stay the same; the
rows assigned to the hidden test set change. If the test score jumps, that is
useful information: this single split was noisy.
"""
)

test_size = st.slider("How much data goes in the test set?", 0.15, 0.50, 0.30, 0.05, key="test_size")
bounce = split_bounce_scores(test_size=test_size, max_seed=10).set_index("seed")
st.line_chart(bounce)
st.caption("The same model bounces because different rows land in the test set.")

st.markdown(
    """
Cross-validation exists because of that bounce. Instead of trusting one split,
it rotates through several hidden chunks and reports the average **and** the
spread. A score without a spread is only half a fact.
"""
)

scores = fold_scores()
fig, ax = ui.figure(7, 3.6)
plot_folds(ax=ax)
ui.show(fig)
score_bits = " + ".join(f"{s:.2f}" for s in scores)
st.metric("5-fold average", f"{scores.mean():.1%}", f"spread ±{scores.std():.1%}")
st.markdown(f"Fold average arithmetic: `({score_bits}) / 5 = {scores.mean():.2f}`.")
st.write([f"{s:.1%}" for s in scores])

baseline = lopsided_baseline()
st.warning(
    f"A useless 'always say the most common class' baseline scores {baseline:.0%} on a "
    "lopsided dataset. Check the baseline first, or you may celebrate a model that learned nothing."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "An honest penguin leaderboard.")

st.markdown(
    """
Here is the penguin race with fair rules: five folds, the same rows for every
model, and mean plus spread. The baseline is included because a fancy model has
to beat the boring answer before it earns applause.
"""
)

leaderboard = cached_leaderboard()
st.dataframe(leaderboard, hide_index=True, use_container_width=True)
st.markdown(
    """
If two means are closer than their spreads, do not brag that one crushed the
other. Read `0.96 ± 0.03` as a small cloud of possible scores, not one magic
number.
"""
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
