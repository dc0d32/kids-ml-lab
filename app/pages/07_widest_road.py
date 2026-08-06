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


@st.cache_data(show_spinner=False)
def _road_animation():
    from kidsml import liftanim

    return liftanim.road_gif_bytes()


@lesson.step("A perfect line can still be nervous", beat="hook")
def _():
    lesson.say(
        """
Chapter 3 showed shapes a straight line cannot split. This chapter adds another
bendy escape: pick the safest road, then let that road curve when it needs to.

A separator line can win and still sweat. Chapter 2's perceptron stops when it finds
**any** line that separates the dots. That is enough for yesterday's dots, but it may
be a nervous choice for tomorrow's dot.

Imagine a new point lands a tiny bit away from where you expected. A line that hugs
one class has no shoulder; one small wiggle can shove the new point across the road.

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
So how wide *can* the road get? Watch it grow. It starts as a thin line hugging the blue
class, then its far edge sweeps across the empty gap.

The road cannot widen forever. The moment each edge reaches the closest dot, the growing
stops — and those dots light up. They are the ones holding the road in place.
"""
    )
    st.image(_road_animation(), caption="The road widens until it jams against the closest points")
    lesson.look_for(
        "the exact moment the growth stops. The edges cannot push past the nearest blue and "
        "red dots, so those dots get ringed. Everything else has room to spare."
    )

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
        why="The road is pinned by support vectors. A far-away dot is back in the grass, not touching the margin.",
        key="ch07_delete_non_support",
    )
    if guess is None:
        return

    model, X, y = fit_linear_svm(remove="non-support")
    fig, ax = lesson.figure(6, 4.8)
    plot_linear_svm_margin(model, X, y, ax=ax, title="removed: non-support point")
    lesson.show(fig)
    lesson.look_for("the road: it is almost the same because the deleted point was not a fence post. It sat far back on its own side, so removing it changed nothing.")


@lesson.step("Delete a fence post", beat="seeit")
def _():
    guess = lesson.predict(
        "Now delete a ringed support vector. What happens?",
        ["The road can jump", "Nothing changes", "All points become support vectors"],
        correct=0,
        why="The old road leaned on that fence post. Pull it out and another point can become the nearest danger.",
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

In these next pictures we use a distance road: each point tugs on nearby space, and the
tug fades as you move away. Grown-ups call this RBF.

`C` is the strictness knob. Low C keeps a wide road even if a few training dots are on the
wrong side. High C makes training mistakes expensive.
"""
    )
    lesson.jargon("RBF", "Short for radial basis function: an SVM road style where nearby points tug more than far-away points.")

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
    lesson.say(
"""
The road style tells the SVM what shapes it is allowed to draw. **Linear** means one
straight road. **Polynomial** makes smooth curvy roads. **RBF** uses distance from
points, so it can make smooth islands.

Put both knobs together and switch the road style. Same data, different idea of what
a road can be.
"""
    )
    lesson.jargon("kernel", "The road style: the rule for what shape the SVM can draw.")

    knobs, picture = lesson.controls()
    with knobs:
        shape = ui.shape_picker(default="circles", key="ch07_style_shape", include=("blobs", "moons", "circles", "xor"))
        noise = ui.noise_slider(default=0.18, key="ch07_style_noise")
        seed = ui.seed_slider(default=2, key="ch07_style_seed")
        kernel = st.selectbox("Kernel: what shape is the road?", ["linear", "poly", "rbf"], index=2, key="ch07_style_kernel")
        C = st.slider("C: care about every training dot", 0.1, 30.0, 3.0, 0.1, key="ch07_style_c")
        gamma = st.slider("gamma: how far each dot reaches", 0.05, 8.0, 1.0, 0.05, key="ch07_style_gamma")

    X, y, model = fit_svm_shape(shape, kernel=kernel, C=C, gamma=gamma, noise=noise, seed=seed)
    with picture:
        fig, ax = lesson.figure(6.2, 5)
        decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False, title=f"{kernel} SVM")
        lesson.show(fig)
        lesson.look_for("where the boundary curls around individual points, especially with high C and high gamma.")
        lesson.aha("Kernel is not a new model here. It is the SVM choosing a different kind of road.")


@lesson.step("Penguins have fence posts too", beat="forreal")
def _():
    lesson.say(
        """
These are the same Palmer penguins from Chapter 04, measured in a different way. Chapter
04 used flipper length and weight; this road uses beak length and beak depth so the picture
still has two axes.

Each row is one penguin, and the target is species. The ringed penguins are the ones close
enough to hold the road in place.
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
graph TD
    A[2D circle dots] --> B[imagine extra features]
    B --> C[straight cut there]
    C --> D[curved road back here]
""",
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
