"""Chapter 14 · Two Layers, Three Neurons."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml.datasets import toy_shape, xor_exact
from kidsml.nn_numpy import MLP
from kidsml.nnplots import (
    boundary_with_hidden,
    hidden_lines,
    hidden_surfaces_figure,
    mlp_snapshot_training,
    model_from_snapshot,
)
from kidsml.plots import decision_boundary, loss_curve

lesson.begin(14)


@st.cache_data(show_spinner=False)
def trained_xor_snapshots():
    X, y = xor_exact()
    snaps = mlp_snapshot_training([2, 3, 1], X, y, lr=0.8, epochs=3000, every=150, activation="tanh", seed=2)
    return X, y, snaps


@st.cache_data(show_spinner=False)
def playground(shape: str, hidden: int, activation: str, lr: float, seed: int):
    X, y = toy_shape(shape, n=180, noise=0.16 if shape != "spiral" else 0.22, seed=seed)
    m = MLP([2, hidden, 1], activation=activation, seed=seed)
    losses = m.fit(X, y, lr=lr, epochs=900, record_every=5)
    return X, y, m, np.array(losses)


@st.cache_data(show_spinner=False)
def overfit_pair():
    X, y = toy_shape("moons", n=70, noise=0.32, seed=10)
    small = MLP([2, 3, 1], activation="tanh", seed=1)
    big = MLP([2, 8, 1], activation="tanh", seed=1)
    small.fit(X, y, lr=0.6, epochs=1200, record_every=20)
    big.fit(X, y, lr=0.6, epochs=2200, record_every=20)
    return X, y, small, big


@lesson.step("XOR is back", beat="hook")
def _():
    lesson.say(
        """
XOR is back because it is the test that tells us whether Part 3 worked.

One neuron cannot solve it: Chapter 3 proved one straight line cannot put opposite corners
together. The escape route was inventing better features.
"""
    )
    lesson.mermaid(
        """
graph LR
    X1[x₁] --> H1[h₁]
    X1 --> H2[h₂]
    X2[x₂] --> H2
    X2 --> H3[h₃]
    H1 --> O[output neuron]
    H2 --> O
    H3 --> O
""",
        height=290,
    )
    lesson.look_for("the middle layer. It invents new reports before the final neuron decides.")


@lesson.step("Make better features by hand", beat="byhand")
def _():
    lesson.say(
        """
We will make two hidden features by hand: **OR-ish** and **AND-ish**. This table is the
whole XOR story in miniature.

In the original `x1, x2` square, the red points sit in opposite corners.
"""
    )
    xor_table = pd.DataFrame(
        {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "OR-ish": [0, 1, 1, 1], "AND-ish": [0, 0, 0, 1], "XOR": [0, 1, 1, 0]}
    )
    st.dataframe(xor_table, hide_index=True, use_container_width=False)
    lesson.look_for("the two red rows. In the new columns, they can land together.")


@lesson.step("Original space versus hidden space", beat="byhand")
def _():
    xor_table = pd.DataFrame(
        {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "OR-ish": [0, 1, 1, 1], "AND-ish": [0, 0, 0, 1], "XOR": [0, 1, 1, 0]}
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    colors = np.where(xor_table["XOR"].to_numpy() == 1, "#EF4444", "#3B82F6")
    axes[0].scatter(xor_table["x1"], xor_table["x2"], c=colors, s=120, edgecolor="white", linewidth=1.5)
    axes[0].set_title("original x₁,x₂ space")
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    axes[0].set_xlim(-0.3, 1.3)
    axes[0].set_ylim(-0.3, 1.3)
    axes[0].set_aspect("equal")
    axes[1].scatter(xor_table["OR-ish"], xor_table["AND-ish"], c=colors, s=120, edgecolor="white", linewidth=1.5)
    h = np.linspace(-0.1, 1.2, 50)
    axes[1].plot(h, (h - 0.5) / 2, color="#111827", linewidth=2)
    axes[1].set_title("new h₁,h₂ space")
    axes[1].set_xlabel("h₁ = OR-ish")
    axes[1].set_ylabel("h₂ = AND-ish")
    axes[1].set_xlim(-0.3, 1.3)
    axes[1].set_ylim(-0.3, 1.3)
    axes[1].set_aspect("equal")
    lesson.show(fig)
    lesson.look_for("the right picture: the red rows are together, and one straight line can separate them.")
    lesson.aha("The hidden layer did not bend the output line. It moved the points into a new space where one straight line works.")


@lesson.step("Predict the learned hidden space", beat="seeit")
def _():
    guess = lesson.predict(
        "After training, can the four XOR points be separated in the hidden layer's new space?",
        ["No, XOR stays impossible", "Yes, the hidden coordinates can make a straight split", "Only if there are ten hidden neurons"],
        correct=1,
        why="The hidden layer invents coordinates. The output neuron then uses one straight split in that new space.",
        key="ch13_hidden_space",
    )
    if guess is None:
        return

    X_xor, y_xor, snaps = trained_xor_snapshots()
    model = model_from_snapshot([2, 3, 1], snaps[-1], activation="tanh", seed=2)
    hidden = model.hidden_outputs(X_xor)
    out = model.predict_proba(X_xor)
    table = pd.DataFrame(
        {
            "x1": X_xor[:, 0],
            "x2": X_xor[:, 1],
            "h1": hidden[:, 0],
            "h2": hidden[:, 1],
            "h3": hidden[:, 2],
            "output": out,
            "XOR": y_xor,
        }
    )
    st.dataframe(table.round(3), hide_index=True, use_container_width=True)
    lesson.look_for("hidden rows with the same XOR answer. They are no longer trapped in opposite corners.")


@lesson.step("Plot the hidden space", beat="seeit")
def _():
    X_xor, y_xor, snaps = trained_xor_snapshots()
    model = model_from_snapshot([2, 3, 1], snaps[-1], activation="tanh", seed=2)
    hidden = model.hidden_outputs(X_xor)
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(y_xor == 1, "#EF4444", "#3B82F6")
    ax.scatter(hidden[:, 0], hidden[:, 1], hidden[:, 2], c=colors, s=90, edgecolor="white", linewidth=1.0)
    ax.set_xlabel("h1")
    ax.set_ylabel("h2")
    ax.set_zlabel("h3")
    ax.set_title("XOR points after the hidden layer")
    lesson.show(fig)
    lesson.look_for("the four points in hidden coordinates. The network changed the map before making the final cut.")


@lesson.step("The three hidden lines", beat="seeit")
def _():
    guess = lesson.predict(
        "Why do the three hidden neurons learn different lines?",
        ["They start with small random differences", "The output neuron orders them by name", "They all get the same gradients forever"],
        correct=0,
        why="Small random starts give slightly different gradients, so the hidden jobs separate.",
        key="ch13_hidden_lines_reason",
    )
    if guess is None:
        return

    X_xor, y_xor, snaps = trained_xor_snapshots()
    model = model_from_snapshot([2, 3, 1], snaps[-1], activation="tanh", seed=2)
    fig = hidden_surfaces_figure(model, X_xor, steps=65)
    lesson.show(fig)
    lesson.look_for("the three straight hidden lines. Each one sends a different ramp reading to the output neuron.")


@lesson.step("The combined boundary", beat="seeit")
def _():
    X_xor, y_xor, snaps = trained_xor_snapshots()
    model = model_from_snapshot([2, 3, 1], snaps[-1], activation="tanh", seed=2)
    fig, ax = lesson.figure(5.4, 4.7)
    boundary_with_hidden(model, X_xor, y_xor, ax=ax, title="Hidden lines plus final boundary", steps=180)
    lesson.show(fig)
    lesson.look_for("hidden lines first, then the shaded final boundary. The bend comes from reading hidden coordinates.")


@lesson.step("Scrub the training", beat="play")
def _():
    lesson.say("Watch the hidden lines slide into place. This is the payoff: the features are learned, not typed by hand.")
    X_xor, y_xor, snaps = trained_xor_snapshots()
    knobs, picture = lesson.controls()
    with knobs:
        step_index = st.slider("Training step to inspect", 0, len(snaps) - 1, len(snaps) - 1, format="%d", key="ch13_training_step")
        st.caption(f"showing step {snaps[step_index]['step']} · loss {snaps[step_index]['loss']:.4f}")
    model = model_from_snapshot([2, 3, 1], snaps[step_index], activation="tanh", seed=2)
    with picture:
        fig, ax = lesson.figure(5.2, 4.5)
        boundary_with_hidden(model, X_xor, y_xor, ax=ax, title="Training snapshot", steps=180)
        lesson.show(fig)
    lesson.look_for("lines that rotate or slide as loss falls. They are becoming useful hidden features.")


@lesson.step("Play with hidden neurons", beat="play")
def _():
    lesson.say("With one hidden neuron you are mostly back to one learned line. Add a few, and the model can invent several features before the final neuron decides.")
    knobs, picture = lesson.controls()
    with knobs:
        shape = st.selectbox("Dataset", ["xor", "moons", "circles", "spiral"], index=0, key="ch13_play_shape")
        hidden = st.slider("Hidden neurons", 1, 8, 3, key="ch13_play_hidden")
        activation = st.selectbox("Activation", ["tanh", "sigmoid", "relu"], index=0, key="ch13_play_activation")
        lr = st.slider("Learning rate", 0.05, 1.5, 0.6, 0.05, key="ch13_play_lr")
        seed = st.slider("Random seed", 0, 10, 3, 1, key="ch13_play_seed")
    X_play, y_play, play_model, _ = playground(shape, hidden, activation, lr, seed)
    with picture:
        fig, ax = lesson.figure(5.0, 4.3)
        decision_boundary(lambda G: play_model.predict_proba(G), X_play, y_play, ax=ax, steps=180, title=play_model.describe())
        if hidden <= 8:
            hidden_lines(ax, play_model, labels=False)
        lesson.show(fig)
    lesson.look_for("how extra hidden lines give the final boundary more ways to bend.")


@lesson.step("Watch the loss curve", beat="play")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        shape = st.selectbox("Loss dataset", ["xor", "moons", "circles", "spiral"], index=0, key="ch13_loss_shape")
        hidden = st.slider("Loss hidden neurons", 1, 8, 3, key="ch13_loss_hidden")
        activation = st.selectbox("Loss activation", ["tanh", "sigmoid", "relu"], index=0, key="ch13_loss_activation")
        lr = st.slider("Loss learning rate", 0.05, 1.5, 0.6, 0.05, key="ch13_loss_lr")
        seed = st.slider("Loss random seed", 0, 10, 3, 1, key="ch13_loss_seed")
    _, _, _, play_losses = playground(shape, hidden, activation, lr, seed)
    with picture:
        fig, ax = lesson.figure(5.0, 4.3)
        loss_curve(play_losses, ax=ax, title="Loss curve")
        lesson.show(fig)
    lesson.look_for("whether loss falls smoothly, stalls, or wiggles. More neurons do not remove the need for learning.")


@lesson.step("More wiggle is a trade", beat="forreal")
def _():
    lesson.say("More hidden neurons give the network more ways to wiggle. That can help with real patterns, and it can over-study noise.")
    X_over, y_over, small, big = overfit_pair()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, m, title in zip(axes, [small, big], ["3 hidden neurons: calmer", "8 hidden neurons: wobblier"]):
        decision_boundary(lambda G, model=m: model.predict_proba(G), X_over, y_over, ax=ax, steps=180, title=title)
    lesson.show(fig)
    lesson.look_for("the wobblier edge. Extra power can fit pattern or noise.")
    lesson.careful("Chapter 15 is about telling helpful wiggle from over-studying.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go make XOR work", beat="challenge")
def _():
    lesson.say(
        """
1. **Smallest XOR solver.** What is the fewest hidden neurons that can solve XOR?
2. **Try spiral.** How many hidden neurons does it need before it looks decent?
3. **Set XOR weights by hand.** Use OR-ish and AND-ish to beat training.
4. **Watch the lines.** Scrub the training slider and say what each hidden line learned.
5. 🧸 **Little Kid Corner:** Three friends make a team. Two notice where you stand. The last friend listens and decides.
"""
    )


lesson.finish()
