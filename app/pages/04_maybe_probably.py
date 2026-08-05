"""Chapter 04 · Maybe, Probably, Definitely."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from kidsml import lesson
from kidsml.datasets import load_table, toy_shape
from kidsml.linear import logistic_proba, sigmoid
from kidsml.nn_numpy import log_loss
from kidsml.plots import ACCENT, COOL, WARM, decision_boundary, draw_line

lesson.begin(4)


@st.cache_data(show_spinner=False)
def penguin_probabilities():
    penguins = load_table("penguins").dropna(subset=["species", "flipper_length_mm", "weight_g"])
    penguins = penguins.copy()
    penguins["is_gentoo"] = (penguins["species"] == "Gentoo").astype(int)
    features = ["flipper_length_mm", "weight_g"]
    model = LogisticRegression().fit(penguins[features], penguins["is_gentoo"])
    probs = model.predict_proba(penguins[features])[:, 1]
    penguins["gentoo_probability"] = probs
    penguins["wrong"] = (probs >= 0.5).astype(int) != penguins["is_gentoo"]
    return penguins, probs


@lesson.step("A boundary should be allowed to shrug", beat="hook")
def _():
    lesson.say(
        """
Chapter 3 bent lines. Now give a straight line a shrug. Some boundaries must
bend, but even when a straight boundary is good enough, Chapter 2's perceptron
has another problem.

It says red or blue and never wavers. A point sitting on the boundary should say:
**honestly, I have no idea**.
"""
    )
    X_shrug, y_shrug = toy_shape("blobs", n=120, noise=0.25, seed=10)
    fig, ax = lesson.figure(5.8, 4.2)
    ax.scatter(
        X_shrug[:, 0],
        X_shrug[:, 1],
        c=np.where(y_shrug == 1, WARM, COOL),
        s=38,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )
    draw_line(2, -2, 0, ax=ax)
    ax.scatter([0], [0], s=190, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
    ax.set_title("The boundary is the unsure place")
    lesson.show(fig)
    lesson.look_for("the circled spot on the line. A hard model must pick a side there, even though it is exactly on the fence.")
    lesson.jargon("logistic regression", "A straight-line score followed by an S-curve that turns it into a probability.")
    lesson.say(
        """
        Keep Chapter 2's raw line score **z**. Positive should lean red. Negative should
        lean blue. Near zero should feel like a shrug.

        So we need a squish machine: any score in, a probability from 0 to 1 out. The one
        used everywhere is **sigmoid(z) = 1 / (1 + e^-z)**. The **e** is a particular
        number, about **2.718**; you do not need to memorize it.
        """
    )
    guess = lesson.predict(
        "The raw score z is exactly 0 on the line. What probability should that become?",
        ["0% red", "50% red", "100% red"],
        correct=1,
        why="At z = 0, sigmoid(z) = 1 / (1 + e⁰) = 1 / (1 + 1) = 0.5. The boundary lands exactly on the shrug!",
        key="ch04_zero",
    )
    if guess is None:
        return

    z = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    p = sigmoid(z)
    st.dataframe({"z": z, "sigmoid(z)": np.round(p, 3)}, hide_index=True, width="stretch")
    lesson.look_for("z = 0 in the middle row. That is the boundary becoming a shrug.")


@lesson.step("The S-curve machine", beat="byhand")
def _():
    lesson.say(
        """
Why this S-shape and not any other? It has the habits we need: every output is
between 0 and 1, 0 turns into exactly 0.5, and opposite scores balance out. For
example, sigmoid(2) is about 0.88, while sigmoid(-2) is about 0.12.
"""
    )
    lesson.mermaid(
        """
flowchart LR
    A[Point features] --> B[Raw score z]
    B --> C[Sigmoid S-curve]
    C --> D[Probability]
    D --> E{p >= 0.5?}
    E -->|yes| F[red]
    E -->|no| G[blue]
""",
        height=260,
    )
    lesson.look_for("the threshold: p = 0.5 happens exactly when the raw score z is 0.")
    lesson.say("It also has a training-friendly meaning: adding 1 to the score multiplies the red-vs-blue odds by the same amount each time.")


@lesson.step("Make the S-curve steeper", beat="seeit")
def _():
    guess = lesson.predict(
        "What happens when the S-curve gets very steep?",
        ["It acts more like a hard red/blue step", "It makes every score 50/50", "It turns into a straight line"],
        correct=0,
        why="Large values shove most scores close to 0 or 1, squeezing the shrug zone into a skinny doorway!",
        key="ch04_steep",
    )
    if guess is None:
        return

    w = st.slider("w: how decisive is the S-curve?", 0.2, 8.0, 1.0, 0.2, key="ch04_steep_w")
    z = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    xs = np.linspace(-6, 6, 300)
    fig, ax = lesson.figure(6.5, 4.4)
    ax.plot(xs, sigmoid(w * xs), color=ACCENT)
    ax.scatter(z, sigmoid(w * z), color=WARM, edgecolors="white", zorder=3)
    ax.axhline(0.5, color="#94A3B8", linestyle="--")
    ax.set_xlabel("line score z")
    ax.set_ylabel("probability of red")
    ax.set_title("The S-curve turns any score into maybe/probably/definitely")
    lesson.show(fig)
    lesson.look_for("the dashed 0.5 line. Scores near zero land near that line, which is the shrug zone.")


@lesson.step("Confidence fades around the line", beat="play")
def _():
    lesson.say(
        """
The probability is curved, but the decision boundary stays straight. The model
says red when **p ≥ 0.5**, and sigmoid reaches 0.5 exactly at **z = 0**.
"""
    )
    X, y = toy_shape("blobs", n=220, noise=0.28, seed=6)
    knobs, picture = lesson.controls()
    with knobs:
        w1 = st.slider("w1", -6.0, 6.0, 2.0, 0.1, key="ch04_w1")
        w2 = st.slider("w2", -6.0, 6.0, 2.0, 0.1, key="ch04_w2")
        b = st.slider("b", -4.0, 4.0, 0.0, 0.1, key="ch04_b")
        prob = logistic_proba(X, w1, w2, b)
        st.metric("Average penalty", f"{log_loss(prob, y):.3f}")
    with picture:
        fig, ax = lesson.figure(6, 5)
        decision_boundary(lambda G: logistic_proba(G, w1, w2, b), X, y, ax=ax, shade_confidence=True)
        draw_line(w1, w2, b, ax=ax)
        ax.set_title("The boundary is straight. The confidence fades near it.")
        lesson.show(fig)
    lesson.look_for("the black boundary staying straight while the shading changes smoothly around it.")


@lesson.step("A probability is a promise", beat="play")
def _():
    lesson.say(
        """
For a concrete score, use **w1 = 2**, **w2 = -1**, **b = 0.5**, and point **(1, 3)**:
**z = 2(1) + (-1)(3) + 0.5 = -0.5**, so **sigmoid(-0.5) ≈ 0.38**.
"""
    )
    penalty = []
    for pred, truth in [(0.9, 1), (0.6, 1), (0.1, 1), (0.99, 0), (0.5, 0)]:
        penalty.append({"predicted red": pred, "true answer": truth, "penalty": round(log_loss([pred], [truth]), 2)})
    st.dataframe(penalty, hide_index=True, width="stretch")
    lesson.look_for("the 0.99 red prediction when the truth is blue. Being certain and wrong is expensive.")
    lesson.careful("Being unsure and wrong is forgivable. Being certain and wrong gets a large penalty.")
    lesson.jargon("log loss", "The penalty score for probability promises. Confident wrong answers cost the most.")


@lesson.step("Penguins near the shrug zone", beat="forreal")
def _():
    guess = lesson.predict(
        "If a real penguin lands closest to 50/50, is that a useful kind of honesty?",
        ["Yes", "No, uncertainty is failure", "Only if the model is perfect"],
        correct=0,
        why="A close call should ring a tiny bell: close call! That warning is useful.",
        key="ch04_penguin_shrug",
    )
    if guess is None:
        return

    penguins, probs = penguin_probabilities()
    uncertain = penguins.iloc[np.argmin(np.abs(probs - 0.5))]
    fig, ax = lesson.figure(7, 5)
    colors = np.where(penguins["is_gentoo"] == 1, WARM, COOL)
    ax.scatter(penguins["flipper_length_mm"], penguins["weight_g"], c=colors, s=32, alpha=0.75, edgecolors="white")
    ax.scatter([uncertain["flipper_length_mm"]], [uncertain["weight_g"]], s=180, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
    ax.set_xlabel("flipper length (mm)")
    ax.set_ylabel("weight (g)")
    ax.set_title("The circled penguin is the model's biggest shrug")
    lesson.show(fig)
    lesson.look_for("the circled penguin sitting closest to the model's fuzzy middle. That penguin is the official vibe check.")
    st.write(f"Most uncertain penguin: **{uncertain['species']}**, probability Gentoo = **{uncertain['gentoo_probability']:.1%}**.")
    lesson.say("Now compare a few probability promises before they get turned into a hard Gentoo/not-Gentoo score.")
    sample = penguins.iloc[[0, len(penguins) // 2, int(np.argmin(np.abs(probs - 0.5))), -1]][["species", "gentoo_probability"]]
    st.bar_chart(sample.set_index("species"))
    lesson.look_for("the bar closest to halfway. That is the model admitting a hard call.")
    score = accuracy_score(penguins["is_gentoo"], probs >= 0.5)
    st.metric("Gentoo/not-Gentoo score", f"{score:.0%}")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    penguins, _ = penguin_probabilities()
    wrong = penguins[penguins["wrong"]]
    wrong_text = "Try changing the features in a notebook if there are no confident mistakes here."
    if len(wrong):
        bad = wrong.iloc[np.argmax(np.abs(wrong["gentoo_probability"] - 0.5))]
        wrong_text = f"One wrong penguin: {bad['species']} at {bad['gentoo_probability']:.1%} Gentoo."
    lesson.say(
        f"""
1. **Find a confident mistake.** {wrong_text}
2. **Crank w.** Make the S-curve a cliff. Does confidence mean correctness?
3. **Find the shrug zone.** Move a point near the boundary and watch the probability approach 50%.
"""
    )
    lesson.kid_corner(
        "Stand near one side of a room and say 'probably red team.' Stand on the middle tape line and say 'I do not know.' "
        "That middle shrug is the new idea."
    )


lesson.finish()
