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
At a party, ask everyone to guess how many jellybeans are in a jar. One person
may be wildly high. Another may be low. The **average** can land closer than
most individual guesses because the high and low mistakes cancel.

But that only works if the guesses are different. If everyone copied the same
wrong number from the same person, averaging would repeat the same mistake.

Forests create useful disagreement on purpose: each tree sees a random sample
of rows and is allowed to consider random columns while it grows.
"""
)

ui.mermaid(
    """
graph LR
    A[training table] --> B[random row samples]
    A --> C[random column choices]
    B --> D[many different trees]
    C --> D
    D --> E[majority vote]
""",
    height=300,
)

st.markdown(
    """
Notice why two trees can disagree even though they came from the same table.
Tree 1 may never see row 17. Tree 2 may not be offered the `speed` column at a
split. Their mistakes become different mistakes, and voting can steady them.
"""
)

ui.little_kid_corner(
    "Ask five people where a hidden toy is. If four point under the couch, check there first. "
    "The crowd vote is stronger than one noisy guess when people are not copying each other."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Crowd trick 1: vote.")

st.markdown(
    """
Here five tiny trees vote red or blue. For point A, the tally is red, red,
blue, red, red: `4 red` versus `1 blue`, so the crowd says red.

Voting helps when errors point in different directions. If one tree overreacts
to a noisy dot and another tree never saw that dot, the majority can ignore the
one odd vote. If all five trees learned the same bad rule, the vote will not save us.
"""
)
st.dataframe(tiny_vote_table(), hide_index=True, use_container_width=True)

st.markdown(
    """
Crowd trick 2 is different. Instead of independent trees voting at the end,
boosting lines trees up in order. Each new tiny tree looks at what the current
team still gets wrong.

The leftover is called a residual: `actual answer - current guess`. If the real
answer is 2 and the current guess is 5, the residual is `2 - 5 = -3`. The next
tree learns to push that guess downward.
"""
)
st.dataframe(tiny_boosting_table(), hide_index=True, use_container_width=True)

st.markdown(
    """
In the table, point C starts at 5 but should be 8, so the leftover is `8 - 5 =
3`. If we add half of that fix, `5 + 1.5 = 6.5`. The new leftover is `8 - 6.5 =
1.5`, smaller than before.
"""
)

ui.jargon("ensemble", "A model made by combining many smaller models.")
ui.jargon("residual", "The leftover mistake: actual answer minus current guess.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "One jagged tree beside a voting forest.")

st.markdown(
    """
Chapter 5 gave us one blocky-bendy tree. Here the forest keeps that same
building block, then lets many versions vote.

The forest is not magic smoothing paint. It is many stair-step boundaries laid
over the same problem, with random differences between them. The final vote can
look calmer than any one tree.
"""
)

shape = ui.shape_picker(default="moons", key="ch06_shape", include=("moons", "circles", "xor", "spiral"))
noise = ui.noise_slider(default=0.25, key="ch06_noise")
seed = ui.seed_slider(default=4, key="ch06_seed")
n_estimators = st.slider("How many trees vote?", 1, 80, 20, 1, key="forest_n")

X, y, tree, forest = fit_tree_and_forest(shape=shape, n_estimators=n_estimators, noise=noise, seed=seed)
fig, axes = ui.two_figures(4.8, 4.2)
decision_boundary(tree.predict, X, y, ax=axes[0], steps=150, shade_confidence=False, title="one tree")
decision_boundary(forest.predict, X, y, ax=axes[1], steps=150, shade_confidence=False, title=f"{n_estimators} trees voting")
ui.show(fig)

st.markdown(
    """
Look at the edges. The single tree often has sharp little bites. The forest is
still made of blocky cuts, but the vote usually removes some lonely mistakes.
"""
)

x1 = st.slider("Click-ish point: feature 1", -2.5, 2.5, 0.0, 0.1, key="vote_x1")
x2 = st.slider("Click-ish point: feature 2", -2.5, 2.5, 0.0, 0.1, key="vote_x2")
votes = forest_vote_counts(forest, [x1, x2])
st.info(f"For that point, **{votes['red']} of {n_estimators}** trees said red and **{votes['blue']}** said blue.")

# ---------------------------------------------------------------------------
ui.beat("play", "Boosting builds a curve from little steps.")

st.markdown(
    """
Boosting is easier to see on one wiggly line. Start with a plain guess. Measure
the leftovers. Fit a small tree to those leftovers. Add a small amount of that
new tree to the running prediction. Then repeat.
"""
)

ui.mermaid(
    """
graph LR
    A[predict] --> B[measure leftovers]
    B --> C[fit tiny tree]
    C --> D[add small fix]
    D --> E[better prediction]
    E --> B
""",
    height=250,
)

st.markdown(
    """
The loop works because the next tree is not trying to relearn the whole answer.
It is learning the part the team still misses. A lot of small corrections can
build a curve that one tiny tree could never draw.
"""
)

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

st.markdown(
    """
Read the three panels left to right. The middle panel is what remains wrong;
the right panel is the newest fix; the left panel is the total after the fixes
have been added.
"""
)

ui.aha("A smooth-looking curve can be built out of many tiny step shapes.")
ui.careful(
    "Boosting overfits more easily than a forest because it keeps staring at the current "
    "mistakes. If some leftovers are noise from bad labels or wiggly data, strong late "
    "trees may chase that noise instead of the real pattern."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "Monsters have a secret rule.")

st.markdown(
    """
The monster table was generated from a secret rule, with 5% of labels flipped
on purpose. That means some training answers are lies. A perfect training score
would be suspicious, because a model would have to learn the lies too.
"""
)

scores, importances, secret = cached_monsters()
st.dataframe(scores, hide_index=True, use_container_width=True)
st.bar_chart(importances.set_index("feature group"))

st.markdown(
    """
Look for the tallest bars before revealing the rule. Attack, magic, and speed
rise to the top; element, home, and height matter much less.
"""
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
