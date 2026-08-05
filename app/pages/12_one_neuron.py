"""Chapter 12 · One Neuron."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from kidsml import lesson
from kidsml.datasets import toy_shape, two_blobs_tiny, xor_exact
from kidsml.nn_numpy import ACTIVATION_BLURB, ACTIVATIONS, Neuron
from kidsml.nnplots import neuron_surface_figure
from kidsml.plots import decision_boundary, draw_line

lesson.begin(12)


@lesson.step("Neural networks are not a new planet", beat="hook")
def _():
    lesson.say(
        """
Part 3 sounds like a new planet: **neural networks**. It is not.

A neuron is Chapter 2's straight-line score bolted to Chapter 4's probability squish.
If you can read `w1*x1 + w2*x2 + b`, you can read the engine inside this circle.

Chapter 11 also showed why the squish matters: straight steps stacked on straight steps
collapse back into one straight step. The squish is what keeps a neuron from being only a
line wearing a costume.
"""
    )
    lesson.mermaid(
        """
graph LR
    X1[x₁] --> M[weights times inputs]
    X2[x₂] --> M
    M --> S((Σ + b))
    S --> A[squish]
    A --> Y[output from 0 to 1]
""",
        height=260,
    )
    lesson.look_for("the weighted sum in the middle. That is the line machine you already built.")
    lesson.aha("One neuron is `output = squish(w1*x1 + w2*x2 + b)`: Chapter 2 inside, Chapter 4 outside.")


@lesson.step("Build three raw scores", beat="byhand")
def _():
    lesson.say(
        """
Use **w1 = 2**, **w2 = -1**, **b = 0.5**. We picked small numbers so every row can be
checked by hand. First build the raw score `z`, then squish it.

The raw score decides which side of the line the point is on. The squish turns distance
from the line into a 0-to-1 confidence gauge.
"""
    )
    hand = pd.DataFrame(
        {
            "x1": [1, 0, 1],
            "x2": [0, 1, 2],
            "z working": ["2*1 + (-1)*0 + 0.5", "2*0 + (-1)*1 + 0.5", "2*1 + (-1)*2 + 0.5"],
            "z": [2.5, -0.5, 0.5],
            "sigmoid(z) approx": [0.92, 0.38, 0.62],
            "prediction": ["red", "blue", "red"],
        }
    )
    st.dataframe(hand, hide_index=True, width="stretch")
    lesson.look_for("the sign of `z`. Positive rows become red; negative rows become blue.")


@lesson.step("What the squish buys", beat="byhand")
def _():
    lesson.say(
        """
Why not leave the raw score alone? For a class answer, `z = 19` and `z = 1900` both mean
red, but a training rule needs a bounded target to compare with `0` and `1`.

The squish clips the wild number into a soft confidence score without moving the fence.
"""
    )
    X, _ = toy_shape("blobs", n=180, noise=0.22, seed=4)
    demo_neuron = Neuron(w=np.array([2.0, -1.0]), b=0.5, activation="sigmoid")
    st.plotly_chart(neuron_surface_figure(demo_neuron, X, steps=45, title="The squish turns the raw ramp into 0..1"), width="stretch")
    lesson.look_for("the ramp flattening into a floor and ceiling. The middle fence stays in the same place.")
    lesson.careful(
        "If you double w1, w2, and b, every raw score doubles. The zero places stay zero, so "
        "the boundary stays put. Far-away points become more confident; the line does not move."
    )


@lesson.step("Raw scores have signs", beat="seeit")
def _():
    lesson.say(
        """
`Neuron.raw(X)` shows the Chapter 2 score before the squish. A positive raw score lands on
one side, a negative score lands on the other, and zero is the fence.

These are toy blob coordinates from two clumps. We chose a few rows from both answers so
the sign flips and the fence is visible.
"""
    )
    X_tiny, y_tiny = two_blobs_tiny()
    # A mix of both answers. The first five rows are all the same class, so the score
    # would never change sign and the fence would be invisible.
    rows = [0, 3, 5, 7, 9]
    neuron = Neuron(w=np.array([1.0, 1.0]), b=-7.0, activation="sigmoid")
    raw = neuron.raw(X_tiny[rows])
    st.code("Neuron(w=[1, 1], b=-7).raw(X)   # x1 + x2 - 7", language="python")
    st.dataframe(
        pd.DataFrame(
            {
                "x1": X_tiny[rows, 0],
                "x2": X_tiny[rows, 1],
                "raw z": raw,
                "side": np.where(raw > 0, "positive", "negative"),
            }
        ),
        hide_index=True,
    )
    lesson.look_for("where the sign flips as you read down. Somewhere between those two rows sits the fence, and the score nearest zero is the point standing closest to it.")


@lesson.step("Move the neuron sliders", beat="play")
def _():
    lesson.say("Drag the learned numbers and watch the green line. That line is where `z = 0`, the fence in the grass.")
    knobs, picture = lesson.controls()
    with knobs:
        w1 = st.slider("w1", -6.0, 6.0, 2.0, 0.2, key="ch12_w1")
        w2 = st.slider("w2", -6.0, 6.0, -1.0, 0.2, key="ch12_w2")
        b = st.slider("b", -4.0, 4.0, 0.5, 0.1, key="ch12_b")
        activation = st.selectbox("Squish", list(ACTIVATIONS), index=list(ACTIVATIONS).index("sigmoid"), key="ch12_activation")
        st.caption(ACTIVATION_BLURB[activation])
    X, y = toy_shape("blobs", n=180, noise=0.22, seed=4)
    play_neuron = Neuron(w=np.array([w1, w2]), b=b, activation=activation)
    with picture:
        fig, ax = lesson.figure(5.0, 4.4)
        decision_boundary(lambda G: play_neuron.forward(G), X, y, ax=ax, steps=180, shade_confidence=True, title="Boundary and confidence")
        draw_line(w1, w2, b, ax=ax)
        lesson.show(fig)
    lesson.look_for("which knob mostly rotates the line, and which knob slides it.")


@lesson.step("Rotate the output surface", beat="play")
def _():
    guess = lesson.predict(
        "Make the weights steeper. What happens to the output ramp?",
        ["It gets flatter", "It changes faster near the fence", "The fence disappears"],
        correct=1,
        why="Bigger weights make raw scores climb faster as your point walks away from `z = 0`. The ramp gets steep!",
        key="ch12_surface_steepness",
    )
    if guess is None:
        return

    knobs, picture = lesson.controls()
    with knobs:
        w1 = st.slider("surface w1", -6.0, 6.0, 2.0, 0.2, key="ch12_surface_w1")
        w2 = st.slider("surface w2", -6.0, 6.0, -1.0, 0.2, key="ch12_surface_w2")
        b = st.slider("surface b", -4.0, 4.0, 0.5, 0.1, key="ch12_surface_b")
        activation = st.selectbox("Surface squish", list(ACTIVATIONS), index=list(ACTIVATIONS).index("sigmoid"), key="ch12_surface_activation")
    X, _ = toy_shape("blobs", n=180, noise=0.22, seed=4)
    play_neuron = Neuron(w=np.array([w1, w2]), b=b, activation=activation)
    with picture:
        st.plotly_chart(neuron_surface_figure(play_neuron, X, steps=45, title="Rotate the ramp"), width="stretch")
    lesson.look_for("the ramp steepness. Far from the fence means a louder answer.")


@lesson.step("Can one neuron do XOR?", beat="play")
def _():
    guess = lesson.predict(
        "Can one neuron solve XOR, where opposite corners need the same colour?",
        ["Yes, if the squish is chosen well", "No, one neuron has one straight boundary", "Yes, if the bias is large"],
        correct=1,
        why="The squish softens confidence, but it never grows a second fence. One neuron brings one straight cut.",
        key="ch12_xor_predict",
    )
    if guess is None:
        return

    knobs, picture = lesson.controls()
    with knobs:
        w1 = st.slider("XOR w1", -6.0, 6.0, 2.0, 0.2, key="ch12_xor_w1")
        w2 = st.slider("XOR w2", -6.0, 6.0, -1.0, 0.2, key="ch12_xor_w2")
        b = st.slider("XOR b", -4.0, 4.0, 0.5, 0.1, key="ch12_xor_b")
    X_xor, y_xor = xor_exact()
    one = Neuron(w=np.array([w1, w2]), b=b, activation="sigmoid")
    fig, ax = lesson.figure(5.0, 4.4)
    decision_boundary(lambda G: one.forward(G), X_xor, y_xor, ax=ax, steps=120, shade_confidence=True, title="One neuron on XOR")
    draw_line(w1, w2, b, ax=ax)
    lesson.show(fig)
    xor_misses = int((one.predict(X_xor) != y_xor).sum())
    st.metric("XOR mistakes (one neuron, still cooked)", xor_misses)
    lesson.look_for("opposite corners. A single straight boundary always cuts the square into two neighbouring chunks.")


@st.cache_data(show_spinner=False)
def fit_blob_models():
    X_fit, y_fit = toy_shape("blobs", n=180, noise=0.18, seed=8)
    mine = Neuron(w=np.zeros(2), b=0.0, activation="sigmoid")
    losses = mine.fit(X_fit, y_fit, lr=0.8, epochs=900)
    sk = LogisticRegression(C=1_000_000, solver="lbfgs").fit(X_fit, y_fit)
    return X_fit, y_fit, mine.w, mine.b, np.array(losses), sk.coef_[0], float(sk.intercept_[0]), float(sk.score(X_fit, y_fit))


@lesson.step("Training chooses the numbers", beat="forreal")
def _():
    lesson.say(
        """
Now we train the neuron instead of choosing its numbers by hand. Scikit-learn calls the same
idea **logistic regression**: a line score, a sigmoid, and a training rule turning the knobs.
Chapter 13 opens that training rule and shows how the knobs move.
"""
    )
    X_fit, y_fit, w_mine, b_mine, _, w_sklearn, b_sklearn, sk_score = fit_blob_models()
    learned = Neuron(w=w_mine, b=b_mine, activation="sigmoid")
    compare = pd.DataFrame(
        {
            "model": ["our Neuron.fit", "sklearn LogisticRegression"],
            "w1": [w_mine[0], w_sklearn[0]],
            "w2": [w_mine[1], w_sklearn[1]],
            "b": [b_mine, b_sklearn],
            "accuracy": [(learned.predict(X_fit) == y_fit).mean(), sk_score],
        }
    )
    st.dataframe(compare.round(3), hide_index=True, width="stretch")
    lesson.look_for("matching behaviour, not matching exact learned numbers.")
    fig, ax = lesson.figure(5.8, 4.6)
    decision_boundary(lambda G: learned.forward(G), X_fit, y_fit, ax=ax, steps=180, title="Our trained neuron")
    lesson.show(fig)
    lesson.look_for("the gap between the two blob clouds. The trained neuron drove a straight divider through that gap.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go bend the neuron", beat="challenge")
def _():
    lesson.say(
        """
1. **Find perfect blob weights.** Use the sliders until the blob mistakes hit zero. Which knob mostly rotates the line?
2. **Say yes to everything.** Make almost the whole plane red. Which bias did it take?
3. **Make a shrug machine.** Set every learned number to zero. What output do you get?
4. **Feel the XOR wall.** Try to get zero XOR mistakes with one neuron, then explain the Chapter 3 reason it cannot happen.
"""
    )

    lesson.kid_corner(
        "Draw a chalk line across the floor. Standing far from the line means a loud, confident answer. Standing right on it means a shrug. Take turns standing somewhere and having the other person shout how sure they are."
    )


lesson.finish()
