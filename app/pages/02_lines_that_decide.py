"""Chapter 02 · Lines That Decide."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import Perceptron

from kidsml import lesson
from kidsml.datasets import toy_shape, two_blobs_tiny
from kidsml.linear import mistake_count, perceptron_history, predict_side, score_line
from kidsml.lineanim import correction_gif_bytes
from kidsml.nn_numpy import perceptron_step
from kidsml.plots import ACCENT, decision_boundary, draw_line, scatter_2d

lesson.begin(2)

X_tiny, y_tiny = two_blobs_tiny()


@st.cache_data(show_spinner=False)
def correction_animation():
    return correction_gif_bytes()

# Five dogs to work through by hand: a mix of both answers, and including the one the
# worked example uses. Taking the first five would have handed the reader five puppies
# and nothing to compare them against.
HAND_ROWS = [0, 3, 5, 7, 9]

w_start = np.array([1.0, 1.0])
b_start = -8.0
w_bad = np.array([1.0, 1.0])
b_bad = -20.0
w_after, b_after, was_wrong = perceptron_step(w_bad, b_bad, X_tiny[5], y_tiny[5])


@lesson.step("A line can decide", beat="hook")
def _():
    lesson.say(
        """
Flip the line from price tag to referee. Chapter 1 used a line to answer **how much?** The line gave a dollar amount.

Same line, new question: **which side?** Use this when the answer has two
buckets: puppy or grown dog, blue or red, yes or no.

The model still computes a number first. That number is a **weighted sum**:
multiply each measurement by its weight, add the bias, then check the sign.
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
    )
    lesson.look_for("the score machine first, then the sign check that turns a number into a colour.")


@lesson.step("Ten dogs at the park", beat="hook")
def _():
    lesson.say(
        """
Here is the thing we want to decide. Ten dogs, and for each one we wrote down
two numbers: **how tall** it is in hand-spans, and **how heavy** it is in bags
of sugar.

Some are puppies. Some are fully grown. Nobody wrote that down — that's the bit
we want the line to work out.
"""
    )

    table = {
        "how tall (x1)": X_tiny[:, 0],
        "how heavy (x2)": X_tiny[:, 1],
        "really a": np.where(y_tiny == 1, "grown dog", "puppy"),
    }
    st.dataframe(table, hide_index=True, width="content")

    lesson.say(
        """
Notice what changed since Chapter 1. There, each thing had **one** number — the
weeks you'd been saving — so the data sat on a number line. Now each dog has
**two** numbers, so every dog is a dot on a **map**.

That's the whole reason this chapter needs a line instead of a threshold. On a
number line you'd split things with a single point. On a map you split them with
a line.
"""
    )

    fig, ax = lesson.figure(5.4, 4.4)
    scatter_2d(X_tiny, y_tiny, ax=ax, size=110)
    ax.set_xlabel("how tall (hand-spans)")
    ax.set_ylabel("how heavy (bags of sugar)")
    ax.set_title("Ten dogs, two measurements each")
    lesson.show(fig)

    lesson.look_for(
        "the empty gap running diagonally between the two clumps. Puppies are small "
        "on both measurements, grown dogs are big on both. Any line you draw through "
        "that gap does the job."
    )


@lesson.step("Guess a line", beat="hook")
def _():
    lesson.say(
        """
So let's guess one. The simplest idea in the world: **add the two numbers
together, and if the total is more than 8, call it a grown dog.**

Written the way the model writes it, that is:

**score = 1·x1 + 1·x2 − 8**

The two **1**s say how much each measurement counts — here, equally. The **−8**
is where we set the bar. Score above zero means the total beat 8.
"""
    )

    fig, ax = lesson.figure(5.4, 4.4)
    scatter_2d(X_tiny, y_tiny, ax=ax, size=110)
    ax.set_xlabel("how tall (hand-spans)")
    ax.set_ylabel("how heavy (bags of sugar)")
    draw_line(w_start[0], w_start[1], b_start, ax=ax, label="x1 + x2 = 8")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Our guess, drawn on the map")
    lesson.show(fig)

    lesson.look_for("that the guessed line lands in the gap. We picked those three numbers by eye — the rest of the chapter is about getting a machine to pick them instead.")

    lesson.jargon(
        "weights and bias",
        "The two multipliers are the <b>weights</b>, and the number on the end is the "
        "<b>bias</b>. Same pair you met in Chapter 1, doing the same jobs — the weights "
        "set the tilt, the bias slides it.",
    )


@lesson.step("Five scores by hand", beat="byhand")
def _():
    lesson.say(
        """
Time to check the guess. Run **score = 1·x1 + 1·x2 − 8** on five selected dogs
and see whether the sign matches the truth. We picked a mix of puppies and grown
dogs, not the first five rows, so the table has both answers.

Take the dog at **(6, 5)**: **1(6) + 1(5) − 8 = 3**. Positive, so the line calls
it a grown dog.
"""
    )
    scores = score_line(X_tiny[HAND_ROWS], w_start[0], w_start[1], b_start)
    hand = {
        "how tall (x1)": X_tiny[HAND_ROWS, 0],
        "how heavy (x2)": X_tiny[HAND_ROWS, 1],
        "score = x1 + x2 − 8": scores,
        "line says": np.where(scores > 0, "grown dog", "puppy"),
        "really a": np.where(y_tiny[HAND_ROWS] == 1, "grown dog", "puppy"),
    }
    st.dataframe(hand, hide_index=True, width="content")
    lesson.look_for(
        "the sign of the score, and nothing else about it. A score of 3 and a score of "
        "300 make the same decision. All the line does is tell you which side you are on."
    )


@lesson.step("One pencil update", beat="byhand")
def _():
    lesson.say(
        """
Now make the line too strict on purpose: **w = (1, 1), b = -20**. The same red
point gets **1(6) + 1(5) - 20 = -9**, so the model guesses blue. Starting with
one clear mistake lets us watch one correction happen.
"""
    )
    guess = lesson.predict(
        "If we add the missed red point to the weights, will this exact point move to the red side?",
        ["Yes, the score will jump positive", "No, one update cannot help", "It will stay exactly tied"],
        correct=0,
        why="The missed point throws its own coordinates onto the weights. That yanks its future score hard toward red!",
        key="ch02_update",
    )
    if guess is None:
        return

    update_table = {
        "line": ["before", "after"],
        "w1": [w_bad[0], w_after[0]],
        "w2": [w_bad[1], w_after[1]],
        "b": [b_bad, b_after],
        "missed point?": [was_wrong, False],
        "score for (6, 5)": [
            score_line(X_tiny[[5]], w_bad[0], w_bad[1], b_bad)[0],
            score_line(X_tiny[[5]], w_after[0], w_after[1], b_after)[0],
        ],
    }
    st.dataframe(update_table, hide_index=True, width="content")
    lesson.say(
        """
        Why does adding the point help? The score for that same point jumps from **-9**
        to **7(6) + 6(5) - 19 = 53**. The point is now strongly on the red side.
        """
    )
    st.image(correction_animation(), caption="One correction: the missed dog flashes, then the line swings round to put it on the red side")
    lesson.look_for(
        "the circled red dog flash first — it is on the wrong side of the strict line up in "
        "the corner. Then the boundary sweeps down until that dog sits on the red side. One "
        "update, one point fixed."
    )
    lesson.say("Here is the same before and after held still, side by side.")
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
    lesson.aha("You trained a model with one pencil correction: find a mistake, nudge the line, check again.")


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


@lesson.step("A real perceptron settles, then stalls", beat="forreal")
def _():
    lesson.say("scikit-learn has a perceptron too. On clean blobs, a straight separator exists, so the model can settle.")
    X, y = toy_shape("blobs", n=180, noise=0.25, seed=2)
    model = Perceptron(max_iter=1000, random_state=0).fit(X, y)
    st.code("Perceptron(max_iter=1000).fit(X, y)", language="python")
    st.write(f"On clean blobs, scikit-learn gets **{model.score(X, y):.0%}** right.")
    guess = lesson.predict(
        "What happens if the correct boundary must bend, but the perceptron only owns one straight line?",
        ["It leaves mistakes behind", "It bends the line", "It refuses to answer"],
        correct=0,
        why="The algorithm can keep fixing one mistake at a time, but a single straight line can't wrap around a curved, moon-shaped group. Some points always end up on the wrong side!",
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
1. **Beat the algorithm, no cap.** Use the sliders until the mistake counter reaches zero.
2. **Set b to 0.** What can the line no longer do? Hint: it must pass through (0, 0).
3. **Make the blobs overlap.** A perceptron only settles when perfection is possible. What does it do instead?
"""
    )
    lesson.kid_corner(
        "Lay a pencil between two piles of toys. One side is red team, the other is blue team. "
        "Move the pencil until nobody is on the wrong side."
    )


lesson.finish()
