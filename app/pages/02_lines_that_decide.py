"""Chapter 02 · Lines That Decide."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import Perceptron

from kidsml import lesson
from kidsml.datasets import toy_shape, two_blobs_tiny
from kidsml.linear import mistake_count, perceptron_history, predict_side, score_line
from kidsml.nn_numpy import perceptron_step
from kidsml.plots import ACCENT, decision_boundary, draw_line, scatter_2d

lesson.begin(2)

X_tiny, y_tiny = two_blobs_tiny()
w_start = np.array([1.0, 1.0])
b_start = -8.0
w_bad = np.array([1.0, 1.0])
b_bad = -20.0
w_after, b_after, was_wrong = perceptron_step(w_bad, b_bad, X_tiny[5], y_tiny[5])


@lesson.step("A line can decide", beat="hook")
def _():
    lesson.say(
        """
Chapter 1 used a line to answer **how much?** The line gave a dollar amount.

Same line, new question: **which side?** A point on one side becomes blue. A
point on the other side becomes red. The model still computes a number first,
but now the sign of that number makes the decision.
"""
    )
    lesson.jargon("perceptron", "An old-school model that decides which side of a line a point is on.")
    lesson.mermaid(
        """
flowchart LR
    A[Point x1 and x2] --> B[Weighted sum z]
    B --> C{z > 0?}
    C -->|yes| D[red]
    C -->|no| E[blue]
""",
        height=240,
    )
    lesson.look_for("the score machine first, then the sign check that turns a number into a colour.")


@lesson.step("Five scores by hand", beat="byhand")
def _():
    lesson.say(
        """
Try this score rule on five points: **score = 1·x1 + 1·x2 - 8**.
Positive score means red. Negative score means blue. For point **(6, 5)** the
arithmetic is **1(6) + 1(5) - 8 = 3**, so the guess is red.
"""
    )
    scores = score_line(X_tiny[:5], w_start[0], w_start[1], b_start)
    hand = {
        "x1": X_tiny[:5, 0],
        "x2": X_tiny[:5, 1],
        "score = x1 + x2 - 8": scores,
        "guess": np.where(scores > 0, "red", "blue"),
        "truth": np.where(y_tiny[:5] == 1, "red", "blue"),
    }
    st.dataframe(hand, hide_index=True, width="stretch")
    lesson.look_for("the sign of the score. Positive and negative are the whole decision.")


@lesson.step("One pencil update", beat="byhand")
def _():
    lesson.say(
        """
Now make the line too strict: **w = (1, 1), b = -20**. The same red point gets
**1(6) + 1(5) - 20 = -9**, so the model guesses blue.
"""
    )
    guess = lesson.predict(
        "If we add the missed red point to the weights, will this exact point move to the red side?",
        ["Yes, the score will jump positive", "No, one update cannot help", "It will stay exactly tied"],
        correct=0,
        why="The point's own coordinates get added to the score rule, so that point pushes hard toward red.",
        key="ch02_update",
    )
    if guess is None:
        return

    st.write(f"Wrong? **{was_wrong}**. New w = **({w_after[0]:.0f}, {w_after[1]:.0f})**, new b = **{b_after:.0f}**.")
    lesson.say(
        """
Why does adding the point help? The score for that same point jumps from **-9**
to **7(6) + 6(5) - 19 = 53**. The point is now strongly on the red side.
"""
    )
    lesson.aha("You trained a model with one pencil correction: find a mistake, nudge the line, check again.")


@lesson.step("Before and after the update", beat="seeit")
def _():
    lesson.say("The circled red point caused the update. Watch how one correction changes the boundary.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, w_now, b_now, title in [
        (axes[0], w_bad, b_bad, "Before the one update"),
        (axes[1], w_after, b_after, "After the one update"),
    ]:
        scatter_2d(X_tiny, y_tiny, ax=ax)
        ax.scatter([X_tiny[5, 0]], [X_tiny[5, 1]], s=180, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
        draw_line(w_now[0], w_now[1], b_now, ax=ax)
        ax.set_title(title)
    lesson.show(fig)
    lesson.look_for("the circled point. The line did not learn a whole rule in one step; it made one wrong point less wrong.")


@lesson.step("Learning is many corrections", beat="seeit")
def _():
    hist = perceptron_history(X_tiny, y_tiny, w=(0, 0), b=0, steps=12)
    step = st.slider("Perceptron learning step", 0, len(hist) - 1, min(4, len(hist) - 1), key="ch02_history_step")
    row = hist.iloc[step]
    fig, ax = lesson.figure(6, 4.6)
    decision_boundary(lambda G: predict_side(G, row.w1, row.w2, row.b), X_tiny, y_tiny, ax=ax, shade_confidence=False)
    draw_line(row.w1, row.w2, row.b, ax=ax)
    ax.set_title(f"Step {step}: {int(row.mistakes)} mistake(s)")
    lesson.show(fig)
    lesson.look_for("the mistake count. Training stops only when a straight line can make every point happy.")


@lesson.step("Move the deciding line", beat="play")
def _():
    lesson.say(
        """
The boundary is the place where **w1·x1 + w2·x2 + b = 0**. The **w** arrow sticks
straight out from that line, toward red.
"""
    )
    X, y = toy_shape("blobs", n=180, noise=0.25, seed=2)
    knobs, picture = lesson.controls()
    with knobs:
        w1 = st.slider("w1", -5.0, 5.0, 1.0, 0.1, key="ch02_w1")
        w2 = st.slider("w2", -5.0, 5.0, 1.0, 0.1, key="ch02_w2")
        b = st.slider("b", -3.0, 3.0, 0.0, 0.1, key="ch02_b")
        st.metric("Mistakes", mistake_count(X, y, w1, w2, b))
    with picture:
        fig, ax = lesson.figure(6, 5)
        decision_boundary(lambda G: predict_side(G, w1, w2, b), X, y, ax=ax, shade_confidence=False)
        draw_line(w1, w2, b, ax=ax)
        ax.arrow(0, 0, w1 * 0.25, w2 * 0.25, color=ACCENT, width=0.025, length_includes_head=True)
        ax.text(w1 * 0.28, w2 * 0.28, "w arrow", color=ACCENT)
        ax.set_title("w points toward red. b slides the line.")
        lesson.show(fig)
    lesson.look_for("b gliding the line in parallel, and w1 or w2 rotating it.")


@lesson.step("A real perceptron settles", beat="forreal")
def _():
    lesson.say("scikit-learn has a perceptron too. On clean blobs, a straight separator exists, so the model can settle.")
    X, y = toy_shape("blobs", n=180, noise=0.25, seed=2)
    model = Perceptron(max_iter=1000, random_state=0).fit(X, y)
    st.code("Perceptron(max_iter=1000).fit(X, y)", language="python")
    st.write(f"On clean blobs, scikit-learn gets **{model.score(X, y):.0%}** right.")


@lesson.step("When a line runs out of road", beat="forreal")
def _():
    guess = lesson.predict(
        "What happens if the correct boundary must bend, but the perceptron only owns one straight line?",
        ["It leaves mistakes behind", "It bends the line", "It refuses to answer"],
        correct=0,
        why="The algorithm keeps fixing mistakes, but one straight line cannot become a curve.",
        key="ch02_bendy",
    )
    if guess is None:
        return

    lesson.say(
        """
A perceptron keeps fixing the first mistake it sees. If the data overlaps, or if
the correct boundary must bend, one fix can undo an earlier fix.
"""
    )
    cols = st.columns(2)
    for col, shape in zip(cols, ["moons", "circles"]):
        Xr, yr = toy_shape(shape, n=220, noise=0.18, seed=4)
        m = Perceptron(max_iter=1000, random_state=0).fit(Xr, yr)
        with col:
            fig, ax = lesson.figure(5.2, 4.5)
            decision_boundary(lambda G, mm=m: mm.predict(G), Xr, yr, ax=ax, shade_confidence=False, title=f"{shape}: {m.score(Xr, yr):.0%} right")
            lesson.show(fig)
    lesson.look_for("the leftover mistakes. The algorithm is not lazy; one straight line has run out of road.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Beat the algorithm.** Use the sliders until the mistake counter reaches zero.
2. **Set b to 0.** What can the line no longer do? Hint: it must pass through (0, 0).
3. **Make the blobs overlap.** A perceptron only settles when perfection is possible. What does it do instead?
"""
    )
    lesson.kid_corner(
        "Lay a pencil between two piles of toys. One side is red team, the other is blue team. "
        "Move the pencil until nobody is on the wrong side."
    )


lesson.finish()
