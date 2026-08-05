"""Chapter 15 · Deeper and Wider."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml.datasets import toy_shape
from kidsml.nn_numpy import MLP, mse
from kidsml.plots import decision_boundary

lesson.begin(15)


@st.cache_resource(show_spinner=False)
def activation_zoo():
    X, y = toy_shape("moons", n=180, noise=0.18, seed=4)
    models = []
    for act in ["sigmoid", "tanh", "relu"]:
        m = MLP([2, 5, 1], activation=act, seed=2)
        m.fit(X, y, lr=0.5 if act != "relu" else 0.08, epochs=1200, record_every=10)
        models.append((act, m))
    return X, y, models


@st.cache_resource(show_spinner=False)
def compare_shapes():
    X, y = toy_shape("spiral", n=170, noise=0.18, seed=6)
    archs = [[2, 3, 1], [2, 10, 1], [2, 5, 5, 1], [2, 5, 5, 5, 1]]
    rows = []
    trained = []
    for sizes in archs:
        m = MLP(sizes, activation="tanh", seed=3)
        losses = m.fit(X, y, lr=0.55, epochs=1600, record_every=20)
        rows.append({"network": " → ".join(map(str, sizes)), "parameters": m.n_parameters(), "final loss": losses[-1]})
        trained.append(m)
    return X, y, pd.DataFrame(rows), trained


@st.cache_resource(show_spinner=False)
def overfit_story(weight_decay: float = 0.0, more_data: bool = False):
    n_train = 240 if more_data else 70
    X, y_clean = toy_shape("spiral", n=n_train, noise=0.28, seed=1)
    rng = np.random.default_rng(4)
    y = y_clean.copy()
    flips = rng.choice(len(y), size=max(1, len(y) // 5), replace=False)
    y[flips] = 1 - y[flips]
    X_test, y_test = toy_shape("spiral", n=240, noise=0.28, seed=99)
    m = MLP([2, 16, 16, 1], activation="tanh", seed=5)
    train_losses, test_losses = [], []
    for e in range(2000):
        m.step(X, y, lr=0.5, weight_decay=weight_decay)
        if e % 25 == 0:
            train_losses.append(mse(m.forward(X), y.reshape(-1, 1)))
            test_losses.append(mse(m.forward(X_test), y_test.reshape(-1, 1)))
    return X, y, X_test, y_test, m, np.array(train_losses), np.array(test_losses)


def draw_losses(train_losses, test_losses, title: str):
    best = int(np.argmin(test_losses))
    fig, ax = lesson.figure(5.2, 4.2)
    ax.plot(train_losses, label="train loss", color="#10B981")
    ax.plot(test_losses, label="test loss", color="#EF4444")
    ax.axvline(best, color="#94A3B8", linestyle="--")
    ax.text(best + 1, test_losses[best], "best test moment", fontsize=9)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    lesson.show(fig)


@lesson.step("Same move, more times", beat="hook")
def _():
    lesson.say(
        """
You have the whole machine now: line scores, squishes, gradients, and hidden features.

Deeper networks do not add a secret ingredient. They stack the same move like pancakes:
make features, squish them, make new features from those features.
"""
    )
    lesson.mermaid(
        """
graph LR
    X[2 inputs] --> H1[5 neurons]
    H1 --> H2[5 neurons]
    H2 --> Y[1 output]
    H1 -. wider .-> H1
    H2 -. deeper .-> Y
""",
        height=260,
    )
    lesson.look_for("the repeated pattern: numbers zip forward, gradients run backward, and hidden layers stack the same move.")
    lesson.say("That buys more flexible boundaries, and it also creates a new danger: memorising noise. Power tools need goggles!")


@lesson.step("Count the learnable numbers", beat="byhand")
def _():
    lesson.say(
        """
Count the learnable numbers in `[2, 5, 5, 1]`. Every arrow is a weight, and every non-input
neuron gets one bias. We are counting the knobs the model can turn.

We picked this size because it has both width (five neurons in a layer) and depth (two
hidden layers), but the arithmetic still fits on one screen.
"""
    )
    counts = pd.DataFrame(
        {
            "layer": ["2 → 5", "5 → 5", "5 → 1", "biases"],
            "working": ["2*5", "5*5", "5*1", "5 + 5 + 1"],
            "count": [10, 25, 5, 11],
        }
    )
    st.dataframe(counts, hide_index=True, width="content")
    lesson.say("That is **40 weights + 11 biases = 51 parameters**. A bigger pile can fit more shapes, including shapes caused by bad luck.")
    lesson.jargon("parameters", "The weights and biases: every adjustable number inside the model.")


@lesson.step("Squishes make different bends", beat="seeit")
def _():
    guess = lesson.predict(
        "Which squish do you expect to make the creasiest boundary?",
        ["sigmoid", "tanh", "ReLU"],
        correct=2,
        why="ReLU is a flat floor glued to a straight ramp, so many ReLUs can build folded-paper boundaries with creases.",
        key="ch15_activation_creases",
    )
    if guess is None:
        return

    X_zoo, y_zoo, models = activation_zoo()
    cols = st.columns(3)
    for col, (act, m) in zip(cols, models):
        with col:
            fig, ax = lesson.figure(4.5, 4.0)
            decision_boundary(lambda G, model=m: model.predict_proba(G), X_zoo, y_zoo, ax=ax, steps=160, title=act)
            lesson.show(fig)
    lesson.look_for("the edges of the coloured regions. Tanh and sigmoid bend like rubber; ReLU often creases like folded paper.")
    lesson.say("Neither style is always best. The squish shape controls the kind of bends the network can build easily.")


@lesson.step("Deeper versus wider", beat="play")
def _():
    lesson.say("On tiny toy shapes, deeper is not automatically better. More capacity means more ways to curve around the points, but training still has to find useful curves.")
    X_cmp, y_cmp, table, trained = compare_shapes()
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    cols = st.columns(4)
    for col, m in zip(cols, trained):
        with col:
            fig, ax = lesson.figure(3.8, 3.5)
            decision_boundary(lambda G, model=m: model.predict_proba(G), X_cmp, y_cmp, ax=ax, steps=140, title=m.describe())
            lesson.show(fig)
    lesson.look_for("the networks that have more parameters but do not automatically draw a cleaner spiral.")
    lesson.say("The big wins for depth show up later, when data has many reusable parts: edges inside images, sounds inside speech, or words inside sentences.")


@lesson.step("Predict over-studying", beat="forreal")
def _():
    lesson.say(
        """
Now watch overfitting. We give the network a small practice set and flip some labels, so
some dots are lies with coordinates.
"""
    )
    guess = lesson.predict(
        "As training continues, what happens to test loss after the broad pattern is learned?",
        ["It keeps falling with train loss", "It turns around and climbs", "It freezes perfectly flat"],
        correct=1,
        why="A high-capacity network can spend its extra wiggles chasing label noise. Practice dots look happier while fresh dots want the calmer rule underneath.",
        key="ch15_test_loss_turn",
    )
    if guess is None:
        return

    lesson.mermaid(
        """
graph LR
    A[broad pattern] --> B[practice loss falls]
    B --> C[tiny wiggles]
    C --> D[train loss lower]
    C --> E[test loss higher]
""",
        height=240,
    )
    lesson.look_for("the split: the same tiny wiggles can help train loss and hurt test loss. Same squiggle, opposite scoreboard.")


@lesson.step("The loss lines split", beat="forreal")
def _():
    X_train, y_train, X_test, y_test, over_model, train_losses, test_losses = overfit_story(0.0, False)
    left, right = st.columns(2)
    with left:
        draw_losses(train_losses, test_losses, "Over-studying: train down, test back up")
    with right:
        fig, ax = lesson.figure(5.2, 4.2)
        decision_boundary(lambda G: over_model.predict_proba(G), X_train, y_train, ax=ax, steps=160, title="Boundary after training")
        lesson.show(fig)
    lesson.look_for("the dashed line. After that point, practice keeps improving while fresh test points get worse.")


@lesson.step("Fix 1: stop at the calm moment", beat="forreal")
def _():
    _, _, _, _, _, train_losses, test_losses = overfit_story(0.0, False)
    draw_losses(train_losses, test_losses, "Early stopping watches the test line")
    lesson.look_for("the lowest point of the red test line, before the network starts chasing weird dots around the yard.")
    lesson.aha("Early stopping works because broad patterns are often learned before noisy details start waving shiny flags.")


@lesson.step("Fix 2: make wiggles expensive", beat="forreal")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        decay = st.slider("Weight decay", 0.0, 0.08, 0.02, 0.01, key="ch15_weight_decay")
    with picture:
        X_train, y_train, _, _, over_model, train_losses, test_losses = overfit_story(decay, False)
        draw_losses(train_losses, test_losses, "Weight decay calms the weights")
        fig, ax = lesson.figure(5.2, 4.2)
        decision_boundary(lambda G: over_model.predict_proba(G), X_train, y_train, ax=ax, steps=160, title="Boundary with weight decay")
        lesson.show(fig)
    lesson.look_for("how stronger decay makes sharp little detours harder to draw.")
    lesson.say("Small weights make gentler ramps, so the boundary has a harder time bending around one weird dot.")


@lesson.step("Fix 3: make one noisy dot less powerful", beat="forreal")
def _():
    guess = lesson.predict(
        "If we add many more honest practice points, what happens to one flipped label?",
        ["It shouts louder", "It matters less", "It deletes the whole pattern"],
        correct=1,
        why="A single noisy point has less power when many honest neighbours surround it.",
        key="ch15_more_data_noise",
    )
    if guess is None:
        return

    X_train, y_train, _, _, over_model, train_losses, test_losses = overfit_story(0.0, True)
    left, right = st.columns(2)
    with left:
        draw_losses(train_losses, test_losses, "More practice questions shrink the gap")
    with right:
        fig, ax = lesson.figure(5.2, 4.2)
        decision_boundary(lambda G: over_model.predict_proba(G), X_train, y_train, ax=ax, steps=160, title="More data after training")
        lesson.show(fig)
    lesson.look_for("the loss gap and the boundary. More data makes single bad labels easier to ignore, like one wrong shout in a stadium.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Smallest spiral net.** Reduce the architecture until the spiral boundary snaps.
2. **Overfit hard; get cooked.** Use few points, many neurons, and no weight decay.
3. **Too much calm.** Raise weight decay until the boundary becomes boring.
4. **More data.** Turn on more practice questions and watch the loss gap shrink.
"""
    )
    lesson.kid_corner("Practice helps. Memorising one worksheet does not. New questions tell the truth, like a surprise quiz with fresh shoes.")


lesson.finish()
