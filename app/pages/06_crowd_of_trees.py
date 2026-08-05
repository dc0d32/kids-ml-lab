"""Chapter 06 · A Crowd of Trees."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from kidsml import lesson, ui
from kidsml.plots import decision_boundary
from kidsml.trees import (
    boosting_trace,
    fit_tree_and_forest,
    forest_vote_counts,
    monster_models,
    tiny_boosting_table,
    tiny_vote_table,
)

lesson.begin(6)


@st.cache_data(show_spinner=False)
def cached_trace(steps, learning_rate, max_depth, seed):
    return boosting_trace(n_steps=steps, learning_rate=learning_rate, max_depth=max_depth, seed=seed)


@st.cache_data(show_spinner=False)
def cached_monsters():
    return monster_models()


@lesson.step("A crowd can be wiser than one guesser", beat="hook")
def _():
    lesson.say(
        """
Shake the jellybean jar and ask a crowd. One person may rocket high. Another
may dive low. The **average** can land closer than
most individual guesses because the high and low mistakes cancel.
"""
    )
    lesson.mermaid(
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
    lesson.look_for("why two trees can disagree even though they came from the same table.")
    lesson.kid_corner(
        "Ask five people where a hidden toy is. If four point under the couch, check there first. "
        "The crowd vote is stronger than one noisy guess when people are not copying each other."
    )


@lesson.step("Crowd trick 1: vote", beat="byhand")
def _():
    lesson.say(
        """
Here five tiny trees vote red or blue. For point A, the tally is red, red, blue,
red, red: `4 red` versus `1 blue`, so the crowd says red.
"""
    )
    st.dataframe(tiny_vote_table(), hide_index=True, width="stretch")
    lesson.look_for("a point where one tree disagrees with the crowd. Voting helps when errors point in different directions.")


@lesson.step("Crowd trick 2: residuals", beat="byhand")
def _():
    lesson.say(
        """
Boosting is different. Instead of independent trees voting at the end, each new
tiny tree looks at what the current team still gets wrong.

The leftover is called a residual: `actual answer - current guess`.
"""
    )
    st.dataframe(tiny_boosting_table(), hide_index=True, width="stretch")
    lesson.look_for("point C. It starts at 5 but should be 8, so the leftover is `8 - 5 = 3`.")
    lesson.jargon("ensemble", "A model made by combining many smaller models.")
    lesson.jargon("residual", "The leftover mistake: actual answer minus current guess.")


@lesson.step("One tree beside a forest", beat="seeit")
def _():
    guess = lesson.predict(
        "If many trees make different small mistakes, what should the forest vote do to lonely bites?",
        ["Often smooth them out", "Make every bite larger", "Copy the first tree exactly"],
        correct=0,
        why="When trees trip in different places, the vote can knock the wobble flat.",
        key="ch06_forest_vote",
    )
    if guess is None:
        return

    shape = ui.shape_picker(default="moons", key="ch06_shape", include=("moons", "circles", "xor", "spiral"))
    noise = ui.noise_slider(default=0.25, key="ch06_noise")
    seed = ui.seed_slider(default=4, key="ch06_seed")
    n_estimators = st.slider("How many trees vote?", 1, 80, 20, 1, key="ch06_forest_n")
    X, y, tree, forest = fit_tree_and_forest(shape=shape, n_estimators=n_estimators, noise=noise, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    decision_boundary(tree.predict, X, y, ax=axes[0], steps=150, shade_confidence=False, title="one tree")
    decision_boundary(forest.predict, X, y, ax=axes[1], steps=150, shade_confidence=False, title=f"{n_estimators} trees voting")
    lesson.show(fig)
    lesson.look_for("the edges. The single tree often has sharp little bites; the forest vote usually calms some of them.")


@lesson.step("Ask the forest how it voted", beat="seeit")
def _():
    lesson.say("A forest can tell you how many trees voted each way for one point.")
    shape = ui.shape_picker(default="moons", key="ch06_vote_shape", include=("moons", "circles", "xor", "spiral"))
    noise = ui.noise_slider(default=0.25, key="ch06_vote_noise")
    seed = ui.seed_slider(default=4, key="ch06_vote_seed")
    n_estimators = st.slider("Trees in the voting crowd", 1, 80, 20, 1, key="ch06_vote_n")
    x1 = st.slider("Click-ish point: feature 1", -2.5, 2.5, 0.0, 0.1, key="ch06_vote_x1")
    x2 = st.slider("Click-ish point: feature 2", -2.5, 2.5, 0.0, 0.1, key="ch06_vote_x2")
    _, _, _, forest = fit_tree_and_forest(shape=shape, n_estimators=n_estimators, noise=noise, seed=seed)
    votes = forest_vote_counts(forest, [x1, x2])
    st.info(f"For that point, **{votes['red']} of {n_estimators}** trees said red and **{votes['blue']}** said blue. Crowd and the crowd, as they say, ate.")


@lesson.step("Boosting is a different crowd trick", beat="play")
def _():
    lesson.say(
        """
Boosting is easier to see on one wiggly line. Start with a plain guess. Measure
the leftovers. Fit a small tree to those leftovers. Add a small amount of that
new tree to the running prediction. Then repeat.
"""
    )
    lesson.mermaid(
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
    lesson.look_for("the loop returning to leftovers. The next tree learns the part the team still misses.")


@lesson.step("The staircase appears", beat="play")
def _():
    guess = lesson.predict(
        "What can many tiny step-shaped fixes build if you add them one at a time?",
        ["A smoother-looking curve", "Only a flat line", "One giant question"],
        correct=0,
        why="Each tiny tree is a wooden stair step. Stack many small steps and the running total can trace a smoother curve!",
        key="ch06_staircase",
    )
    if guess is None:
        return

    steps = st.slider("Boosting step", 1, 50, 12, key="ch06_boost_step")
    rate = st.slider("How much of each fix to add", 0.05, 0.60, 0.25, 0.05, key="ch06_boost_rate")
    stump_depth = st.slider("How deep is each tiny tree?", 1, 4, 1, key="ch06_boost_depth")
    trace = cached_trace(50, rate, stump_depth, 4)
    stage = trace["stages"][steps - 1]
    fig, ax = lesson.figure(7, 4.2)
    ax.scatter(trace["x"], trace["y"], color="#3B82F6", edgecolors="white", label="data")
    ax.plot(trace["x_grid"], stage["running_grid"], color="#10B981")
    ax.set_title("running prediction")
    lesson.show(fig)
    lesson.look_for("the green staircase getting closer to the blue dots as the step slider grows.")


@lesson.step("Leftover, newest fix, total", beat="play")
def _():
    steps = st.slider("Boosting step for the three panels", 1, 50, 12, key="ch06_boost_panels_step")
    rate = st.slider("Fix size for the three panels", 0.05, 0.60, 0.25, 0.05, key="ch06_boost_panels_rate")
    stump_depth = st.slider("Tiny tree depth for the three panels", 1, 4, 1, key="ch06_boost_panels_depth")
    trace = cached_trace(50, rate, stump_depth, 4)
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
    lesson.show(fig)
    lesson.look_for("the middle panel first, then the newest fix, then the total after that fix is added.")
    lesson.aha("A smooth-looking curve can be built out of many tiny step shapes.")


@lesson.step("Late fixes can chase noise", beat="play")
def _():
    lesson.careful(
        "Boosting overfits more easily than a forest because it keeps staring at the current mistakes. "
        "If some leftovers are noise from bad labels or wiggly data, strong late trees may chase that noise instead of the real pattern."
    )


@lesson.step("Monsters have a secret rule", beat="forreal")
def _():
    guess = lesson.predict(
        "Before revealing the rule, which feature group do you expect to matter most?",
        ["attack", "home", "height_cm", "element"],
        correct=0,
        why="Attack grabs the tallest bar, with magic and speed right behind it.",
        key="ch06_monsters",
    )
    if guess is None:
        return

    scores, importances, secret = cached_monsters()
    st.dataframe(scores, hide_index=True, width="stretch")
    st.bar_chart(importances.set_index("feature group"))
    lesson.look_for("the tallest bars before reading the rule. Attack, magic, and speed rise to the top.")
    st.code(secret)
    st.warning("Five percent of the labels were flipped on purpose. A model scoring 100% here would be suspicious.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. How few trees does the forest need before it beats the one tree?
2. Make boosting overfit the wiggle: use many steps and deeper tiny trees. Where do the wobbles appear?
3. Find a monster feature the model ignores. Does the secret rule agree?
"""
    )
    lesson.kid_corner("Guess jellybeans with a group. Compare one guess with the group average.")


lesson.finish()
