"""Chapter 15 · Two Layers, Three Neurons."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kidsml import foldspace, lesson
from kidsml.datasets import toy_shape
from kidsml.nn_numpy import MLP
from kidsml.nnplots import (
    boundary_with_hidden,
    hidden_lines,
    hidden_surfaces_figure,
    model_from_snapshot,
)
from kidsml.plots import COOL, PANEL, WARM, decision_boundary, loss_curve

lesson.begin(15)


@st.cache_data(show_spinner=False)
def trained_xor_snapshots(lr: float = 0.25):
    return foldspace.learned_xor_snapshots(lr=lr, n=40)


@st.cache_data(show_spinner=False)
def fold_animation():
    return foldspace.fold_gif_bytes()


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
XOR is back, the tiny checkerboard that keeps catching one-neuron models in the act.

One neuron cannot solve it: Chapter 3 proved one straight line cannot put opposite corners
together. The escape route was inventing better features.
"""
    )
    lesson.mermaid(
        """
graph LR
    X1[x1] --> H1[h1]
    X1 --> H2[h2]
    X1 --> H3[h3]
    X2[x2] --> H1
    X2 --> H2
    X2 --> H3
    H1 --> O[output neuron]
    H2 --> O
    H3 --> O
""",
    )
    lesson.look_for("the middle layer. It invents new reports before the final neuron makes the call.")
    lesson.jargon("hidden layer", "A layer between the inputs and the final output. You see its numbers, but they are not the final answer.")
    lesson.say("The names `h₁`, `h₂`, and `h₃` mean hidden neuron 1, 2, and 3. Each one outputs a new number for the final neuron to read.")


@lesson.step("Make better features by hand", beat="byhand")
def _():
    lesson.say(
        """
We will make two hidden features by hand: **OR-ish** and **AND-ish**. This table is the
whole XOR story shrunk to four dots.

In the original `x1, x2` square, the red points sit in opposite corners. In the new columns, OR-ish is `h1` and AND-ish is `h2`: two fresh coordinates we invented from the same dot.
"""
    )
    xor_table = pd.DataFrame(
        {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "OR-ish": [0, 1, 1, 1], "AND-ish": [0, 0, 0, 1], "XOR": [0, 1, 1, 0]}
    )
    st.dataframe(xor_table, hide_index=True, width="content")
    lesson.say(
        """
Here is the output line in the new space: `score = OR-ish - 2*AND-ish - 0.5`.
For a red row, `(OR-ish, AND-ish) = (1, 0)`, so `1 - 2*0 - 0.5 = 0.5`.
For the blue `(1, 1)` row, `1 - 2*1 - 0.5 = -1.5`.
"""
    )
    lesson.look_for("the two red rows. In the original square they sit in opposite corners, but in the OR-ish and AND-ish columns they both read (1, 0) — the same spot.")


@lesson.step("Watch the fold happen", beat="byhand")
def _():
    lesson.say(
        """
Those two new columns are more than numbers in a table. They *move the four dots*. Here is
the move, played out.

Each dot starts at its `(x1, x2)` spot and slides to its `(OR-ish, AND-ish)` spot from the
table. The blue `(0, 0)` corner stays at `(0, 0)`. The blue `(1, 1)` corner stays at
`(1, 1)`. But both red corners — `(0, 1)` and `(1, 0)` — head for the very same place,
`(1, 0)`.
"""
    )
    st.image(fold_animation(), caption="XOR folding from the original square into the hidden space")
    lesson.look_for(
        "the two red corners travelling toward each other until they sit side by side near (1, 0), while the two blue corners stay put and far apart. At the end one straight green line drops in with both reds on one side and both blues on the other."
    )
    lesson.say("Prefer a still picture? Here is the same thing frozen: before on the left, after on the right.")
    lesson.show(foldspace.fold_still_pair())
    lesson.aha(
        "The hidden layer never bent the output line. It moved the points into a new space where one straight line already works. Same ruler, better map."
    )


@lesson.step("Predict the learned hidden space", beat="seeit")
def _():
    guess = lesson.predict(
        "After training, can the four XOR points be separated in the hidden layer's new space?",
        ["No, XOR stays impossible", "Yes, the hidden coordinates can make a straight split", "Only if there are ten hidden neurons"],
        correct=1,
        why="The hidden layer invents coordinates. The output neuron then uses one straight split in that new space.",
        key="ch15_hidden_space",
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
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    lesson.look_for("the h1, h2, h3 columns. They are the three hidden-neuron outputs: new coordinates for the same four XOR dots.")
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(y_xor == 1, WARM, COOL)
    ax.scatter(hidden[:, 0], hidden[:, 1], hidden[:, 2], c=colors, s=90, edgecolor=PANEL, linewidth=1.0)
    ax.set_xlabel("h1 output")
    ax.set_ylabel("h2 output")
    ax.set_zlabel("h3 output")
    ax.set_title("XOR points after the hidden layer")
    lesson.show(fig)
    lesson.look_for("the hidden-space picture. The table numbers became a new map where the final neuron can make one clean cut.")


@lesson.step("Hidden lines make the bend", beat="seeit")
def _():
    guess = lesson.predict(
        "Why do the three hidden neurons learn different lines?",
        ["They start with small random differences", "The output neuron orders them by name", "They all get the same gradients forever"],
        correct=0,
        why="Small random starts give slightly different gradients, so the hidden jobs peel apart.",
        key="ch15_hidden_lines_reason",
    )
    if guess is None:
        return

    X_xor, y_xor, snaps = trained_xor_snapshots()
    model = model_from_snapshot([2, 3, 1], snaps[-1], activation="tanh", seed=2)
    lesson.say("Each hidden neuron draws one straight line and sends a ramp reading. The output neuron adds those readings up, so the final edge bends in the original picture.")
    left, right = st.columns(2, gap="large")
    with left:
        fig = hidden_surfaces_figure(model, X_xor, steps=65)
        lesson.show(fig)
    with right:
        fig, ax = lesson.figure(5.4, 4.7)
        boundary_with_hidden(model, X_xor, y_xor, ax=ax, title="Hidden lines plus final boundary", steps=180)
        lesson.show(fig)
    lesson.look_for("the left picture holds the straight hidden lines. The right picture holds the final boundary, and it is bent. Straight lines in the hidden space become one curved edge back in the original picture.")


@lesson.step("Scrub the training", beat="play")
def _():
    lesson.say("Nobody typed OR-ish and AND-ish. The network **learned** the hidden lines by training. Drag the slider to replay that training from the first step to the last.")
    knobs, picture = lesson.controls()
    with knobs:
        lr = st.slider(
            "Learning rate", 0.1, 0.6, 0.25, 0.05, key="ch15_scrub_lr",
            help="How big a step the network takes each time. Bigger is faster but jumpier.",
        )
        X_xor, y_xor, snaps = trained_xor_snapshots(lr)
        step_index = st.slider("Training step to inspect", 0, len(snaps) - 1, 0, format="%d", key="ch15_training_step")
        st.caption(f"showing step {snaps[step_index]['step']} · loss {snaps[step_index]['loss']:.4f}")
    model = model_from_snapshot([2, 3, 1], snaps[step_index], activation="tanh", seed=2)
    with picture:
        fig, ax = lesson.figure(5.2, 4.5)
        boundary_with_hidden(model, X_xor, y_xor, ax=ax, title="Training snapshot", steps=180)
        lesson.show(fig)
    lesson.look_for("the hidden lines rotating and sliding as the loss drops. Turn the learning rate down and the change spreads out across the whole slider; turn it up and almost all of it happens in the first tug.")


@lesson.step("Play with hidden neurons", beat="play")
def _():
    lesson.say("With one hidden neuron you are mostly back to one learned line. Add a few, and the model can invent several features before the final neuron decides.")
    knobs, picture = lesson.controls()
    with knobs:
        shape = st.selectbox("Dataset", ["xor", "moons", "circles", "spiral"], index=0, key="ch15_play_shape")
        hidden = st.slider("Hidden neurons", 1, 8, 3, key="ch15_play_hidden")
        activation = st.selectbox("Activation", ["tanh", "sigmoid", "relu"], index=0, key="ch15_play_activation")
        lr = st.slider("Learning rate", 0.05, 1.5, 0.6, 0.05, key="ch15_play_lr")
        seed = st.slider("Random seed", 0, 10, 3, 1, key="ch15_play_seed")
    X_play, y_play, play_model, _ = playground(shape, hidden, activation, lr, seed)
    with picture:
        fig, ax = lesson.figure(5.0, 4.3)
        decision_boundary(lambda G: play_model.predict_proba(G), X_play, y_play, ax=ax, steps=180, title=play_model.describe())
        if hidden <= 8:
            hidden_lines(ax, play_model, labels=False)
        lesson.show(fig)
    lesson.look_for("how extra hidden lines give the final boundary more elbows and corners to work with.")


@lesson.step("Watch the loss curve", beat="play")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        shape = st.selectbox("Loss dataset", ["xor", "moons", "circles", "spiral"], index=0, key="ch15_loss_shape")
        hidden = st.slider("Loss hidden neurons", 1, 8, 3, key="ch15_loss_hidden")
        activation = st.selectbox("Loss activation", ["tanh", "sigmoid", "relu"], index=0, key="ch15_loss_activation")
        lr = st.slider("Loss learning rate", 0.05, 1.5, 0.6, 0.05, key="ch15_loss_lr")
        seed = st.slider("Loss random seed", 0, 10, 3, 1, key="ch15_loss_seed")
    _, _, _, play_losses = playground(shape, hidden, activation, lr, seed)
    with picture:
        fig, ax = lesson.figure(5.0, 4.3)
        loss_curve(play_losses, ax=ax, title="Loss curve")
        lesson.show(fig)
    lesson.look_for("whether loss falls smoothly, stalls, or wiggles. More neurons do not remove the need for learning.")


@lesson.step("More neurons, more bends", beat="forreal")
def _():
    lesson.say(
        """
Every hidden neuron adds one more bend the boundary is allowed to make. So the real
question is: how many bends do you actually want?

Here are two networks on the same noisy moons. The left one has 3 hidden neurons, so 3
bends to work with. The right one has 8, so 8 bends. Watch what the extra five do.
"""
    )
    X_over, y_over, small, big = overfit_pair()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, m, title in zip(axes, [small, big], ["3 hidden neurons: 3 bends", "8 hidden neurons: 8 bends"]):
        decision_boundary(lambda G, model=m: model.predict_proba(G), X_over, y_over, ax=ax, steps=180, title=title)
    lesson.show(fig)
    lesson.look_for("the right edge. Those extra bends let it curl around single stray dots — that is the network fitting the noise instead of the pattern the two moons share.")
    lesson.careful("More bends can trace a real curve, and more bends can trace noise. Chapter 16 is about telling those two apart before the boundary gets dramatic.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go make XOR work", beat="challenge")
def _():
    lesson.say(
        """
1. **Smallest XOR solver.** What is the fewest hidden neurons that can solve XOR?
2. **Try spiral.** How many hidden neurons does it need before it starts curling the right way?
3. **Set XOR weights by hand.** Use OR-ish and AND-ish to beat training.
4. **Read the hidden lines.** Scrub the training slider and say what each hidden line learned.
"""
    )

    lesson.kid_corner(
        "Three friends, one job. Two of them each watch one thing: *are you left of the tree?* and *are you past the bench?* Neither knows the answer alone. The third friend hears both replies and calls it. That is a hidden layer: little reports become one decision."
    )


lesson.finish()
