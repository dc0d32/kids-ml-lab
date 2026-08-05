"""Chapter 07 · The Widest Road."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import lesson, ui
from kidsml.plots import decision_boundary, scatter_2d
from kidsml.trees import (
    fit_circles_lifted,
    fit_linear_svm,
    fit_svm_shape,
    penguin_svm,
    plot_linear_svm_margin,
    svm_hand_points,
)

lesson.begin(7)


@lesson.step("A perfect line can still be nervous", beat="hook")
def _():
    lesson.say(
        """
Chapter 2's perceptron stops when it finds **any** line that separates the dots. That is
    enough for yesterday's dots, but it may be a nervous choice for tomorrow's dot.

Your instinct says: pick the line with the biggest empty gap around it. Wider roads survive
small surprises better.
"""
    )

    X_hand, y_hand, _ = svm_hand_points()
    fig, ax = lesson.figure(6, 4.5)
    scatter_2d(X_hand, y_hand, ax=ax)
    ax.axvline(2.25, color="#94A3B8", linewidth=2, label="hugs blue")
    ax.axvline(3.75, color="#94A3B8", linewidth=2, linestyle="--", label="hugs red")
    ax.axvline(3.0, color="#10B981", linewidth=3, label="middle road")
    ax.legend(fontsize=8)
    lesson.show(fig)
    lesson.look_for("the green road: it leaves room on both sides instead of hugging one class.")

    lesson.kid_corner(
        "Imagine walking between two puddles. You do not walk touching one puddle. "
        "You take the widest dry path because your foot might wobble."
    )


@lesson.step("Measure the safety gap", beat="byhand")
def _():
    lesson.say(
        """
The safety gap is the distance from the road to the closest dot on either side. A road is
only as safe as its closest danger.

For `x = 2.5`, the nearest blue dot has `x = 2`, so the blue gap is `2.5 - 2 = 0.5`.
The nearest red dot has `x = 4`, so the red gap is `4 - 2.5 = 1.5`.
"""
    )

    X_hand, _, candidates = svm_hand_points()
    st.dataframe(
        pd.DataFrame({"x1": X_hand[:, 0], "x2": X_hand[:, 1], "class": ["blue"] * 3 + ["red"] * 3}),
        hide_index=True,
    )
    st.dataframe(candidates, hide_index=True, width="stretch")
    lesson.look_for("the `smallest gap` column. The widest road is the one whose worst danger is safest.")

    lesson.aha("Both roads separate the dots. The one with the bigger smallest gap wins.")
    lesson.jargon("margin", "The empty road between the two classes.")


@lesson.step("A few points hold the road", beat="seeit")
def _():
    lesson.say(
        """
Once the road is as wide as possible, most dots are not pushing on it. The closest dots
touch the edge of the road like fence posts.
"""
    )

    model, X, y = fit_linear_svm(remove="none")
    fig, ax = lesson.figure(6, 4.8)
    plot_linear_svm_margin(model, X, y, ax=ax, title="support vectors hold the margin")
    lesson.show(fig)
    lesson.look_for("the ringed points. Those are the dots the widest road is resting against.")

    lesson.jargon("support vector machine", "A model that chooses the separating road with the widest safe gap.")
    lesson.jargon("support vectors", "The closest points that hold the road in place.")


@lesson.step("Delete a far-away point", beat="seeit")
def _():
    guess = lesson.predict(
        "Delete a point far from the boundary. What happens to the road?",
        ["It shifts a lot", "It does not move", "It gets much wider"],
        correct=1,
        why="Only the support vectors hold the road in place. A far-away dot is not touching the margin.",
        key="ch07_delete_non_support",
    )
    if guess is None:
        return

    model, X, y = fit_linear_svm(remove="non-support")
    fig, ax = lesson.figure(6, 4.8)
    plot_linear_svm_margin(model, X, y, ax=ax, title="removed: non-support point")
    lesson.show(fig)
    lesson.look_for("the road: it is almost the same because the deleted point was not a fence post.")


@lesson.step("Delete a fence post", beat="seeit")
def _():
    guess = lesson.predict(
        "Now delete a ringed support vector. What happens?",
        ["The road can jump", "Nothing changes", "All points become support vectors"],
        correct=0,
        why="The old road was resting against that point. Remove it and a different point may become the closest danger.",
        key="ch07_delete_support",
    )
    if guess is None:
        return

    model, X, y = fit_linear_svm(remove="support")
    fig, ax = lesson.figure(6, 4.8)
    plot_linear_svm_margin(model, X, y, ax=ax, title="removed: support vector")
    lesson.show(fig)
    lesson.look_for("which points are ringed now. The fence posts changed, so the road changed too.")


@lesson.step("C is the strictness knob", beat="play")
def _():
    lesson.say(
        """
Real data is messy, so the road sometimes has to choose: stay wide, or bend hard to fix
every training dot.

`C` is the strictness knob. Low C keeps a wide road even if a few training dots are on the
wrong side. High C makes training mistakes expensive.
"""
    )

    knobs, picture = lesson.controls()
    with knobs:
        shape = ui.shape_picker(default="moons", key="ch07_c_shape", include=("blobs", "moons", "circles", "xor"))
        noise = ui.noise_slider(default=0.22, key="ch07_c_noise")
        seed = ui.seed_slider(default=2, key="ch07_c_seed")
        C = st.slider("C: care about every training dot", 0.1, 30.0, 3.0, 0.1, key="ch07_c_value")

    X, y, model = fit_svm_shape(shape, kernel="rbf", C=C, gamma=1.0, noise=noise, seed=seed)
    with picture:
        fig, ax = lesson.figure(6.2, 5)
        decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False, title="RBF SVM, changing C")
        lesson.show(fig)
    lesson.look_for("noisy dots near the boundary. High C tries harder to satisfy them.")


@lesson.step("Gamma is the reach knob", beat="play")
def _():
    lesson.say(
        """
`gamma` is the reach knob for the RBF road. Low gamma means each point reaches far, making
broad smooth shapes.

High gamma means each point reaches a short distance, which can create tiny islands.
"""
    )

    knobs, picture = lesson.controls()
    with knobs:
        shape = ui.shape_picker(default="circles", key="ch07_gamma_shape", include=("blobs", "moons", "circles", "xor"))
        noise = ui.noise_slider(default=0.18, key="ch07_gamma_noise")
        seed = ui.seed_slider(default=2, key="ch07_gamma_seed")
        gamma = st.slider("gamma: how far each dot reaches", 0.05, 8.0, 1.0, 0.05, key="ch07_gamma_value")

    X, y, model = fit_svm_shape(shape, kernel="rbf", C=3.0, gamma=gamma, noise=noise, seed=seed)
    with picture:
        fig, ax = lesson.figure(6.2, 5)
        decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False, title="RBF SVM, changing gamma")
        lesson.show(fig)
    lesson.look_for("tiny islands when gamma is high. Short reach can turn memorisation into a picture.")
    lesson.careful("A kid-repeat version: **C is strictness; gamma is reach**. Strict and short-reach can memorise islands.")


@lesson.step("Pick the road style", beat="play")
def _():
    lesson.say("Now put both knobs together and switch the road style. Same data, different idea of what a road can be.")

    knobs, picture = lesson.controls()
    with knobs:
        shape = ui.shape_picker(default="circles", key="ch07_style_shape", include=("blobs", "moons", "circles", "xor"))
        noise = ui.noise_slider(default=0.18, key="ch07_style_noise")
        seed = ui.seed_slider(default=2, key="ch07_style_seed")
        kernel = st.selectbox("Road style", ["linear", "poly", "rbf"], index=2, key="ch07_style_kernel")
        C = st.slider("C: care about every training dot", 0.1, 30.0, 3.0, 0.1, key="ch07_style_c")
        gamma = st.slider("gamma: how far each dot reaches", 0.05, 8.0, 1.0, 0.05, key="ch07_style_gamma")

    X, y, model = fit_svm_shape(shape, kernel=kernel, C=C, gamma=gamma, noise=noise, seed=seed)
    with picture:
        fig, ax = lesson.figure(6.2, 5)
        decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False, title=f"{kernel} SVM")
        lesson.show(fig)
    lesson.look_for("where the boundary curls around individual points, especially with high C and high gamma.")


@lesson.step("Penguins have fence posts too", beat="forreal")
def _():
    lesson.say(
        """
Penguins are real data, not toy dots. We use two beak measurements so the road can be drawn.
The ringed penguins are the ones close enough to hold the road in place.
"""
    )

    X_peng, y_peng, _, peng_model, support = penguin_svm()
    fig, ax = lesson.figure(6, 4.8)
    decision_boundary(
        peng_model.predict,
        X_peng,
        y_peng,
        ax=ax,
        steps=150,
        shade_confidence=False,
        title="penguin species from two beak measurements",
    )
    ax.scatter(support[:, 0], support[:, 1], s=95, facecolors="none", edgecolors="black", linewidths=1.4)
    ax.set_xlabel("beak length (mm)")
    ax.set_ylabel("beak depth (mm)")
    lesson.show(fig)
    lesson.look_for("how many penguins are not ringed. They are real data, but they do not set the final road width.")
    st.info(f"This model depends on {len(support)} support vectors out of {len(X_peng)} penguins.")


@lesson.step("The kernel trick in kid language", beat="forreal")
def _():
    lesson.say(
        """
Chapter 3 invented a `radius²` feature so circles could be cut by a straight slice in a
lifted space. An RBF SVM uses the same kind of idea without asking you to build all those
extra columns by hand.
"""
    )

    lesson.mermaid(
        """
graph LR
    A[2D circle dots] --> B[imagine extra features]
    B --> C[straight cut there]
    C --> D[curved road back here]
""",
        height=220,
    )
    lesson.look_for("the detour: make the data easier somewhere else, then read the cut back here.")

    X_c, y_c, lifted_predict = fit_circles_lifted()
    X_rbf, y_rbf, rbf_model = fit_svm_shape("circles", kernel="rbf", C=2.0, gamma=1.0, noise=0.12, seed=2)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    decision_boundary(lifted_predict, X_c, y_c, ax=axes[0], steps=140, shade_confidence=False, title="linear SVM + invented radius feature")
    decision_boundary(rbf_model.predict, X_rbf, y_rbf, ax=axes[1], steps=140, shade_confidence=False, title="RBF SVM shortcut")
    lesson.show(fig)
    lesson.look_for("the same lesson in both panels: a straight idea elsewhere becomes a curved boundary here.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break the road", beat="challenge")
def _():
    lesson.say(
        """
1. Add noise and find the C where the road starts chasing it.
2. Make gamma so large that the RBF SVM memorises islands.
3. On circles, beat the RBF SVM with a linear SVM plus your own `x1² + x2²` feature.
4. 🧸 **Little Kid Corner:** Put two sticker colours on a table. Draw the widest road between them.
"""
    )


lesson.finish()
