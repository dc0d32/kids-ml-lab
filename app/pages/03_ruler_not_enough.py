"""Chapter 03 · When a Ruler Isn't Enough."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

from kidsml import lesson, ui
from kidsml.datasets import toy_shape, xor_exact
from kidsml.linear import predict_side
from kidsml.plots import ACCENT, COOL, WARM, decision_boundary, draw_line, scatter_2d

lesson.begin(3)

X_xor, y_xor = xor_exact()


@lesson.step("Let the ruler fail", beat="hook")
def _():
    lesson.say(
        """
Chapter 2 gave us a ruler: one straight line can choose red or blue. Now the line
runs out of road.

A straight ruler is about to lose a fight with a circle. The middle wants one
answer and the ring wants the other.
"""
    )
    X_fail, y_fail = toy_shape("circles", n=160, noise=0.08, seed=0)
    knobs, picture = lesson.controls()
    with knobs:
        w1_fail = st.slider("circle w1", -5.0, 5.0, 1.0, 0.2, key="ch03_circle_w1")
        w2_fail = st.slider("circle w2", -5.0, 5.0, 0.0, 0.2, key="ch03_circle_w2")
        b_fail = st.slider("circle b", -3.0, 3.0, 0.0, 0.2, key="ch03_circle_b")
        mistakes = int((predict_side(X_fail, w1_fail, w2_fail, b_fail) != y_fail).sum())
        st.metric("Circle mistakes", mistakes)
    with picture:
        fig, ax = lesson.figure(6, 4.6)
        decision_boundary(lambda G: predict_side(G, w1_fail, w2_fail, b_fail), X_fail, y_fail, ax=ax, shade_confidence=False)
        draw_line(w1_fail, w2_fail, b_fail, ax=ax)
        ax.set_title("Try to make circles perfect with one line")
        lesson.show(fig)
    lesson.look_for("the best-looking ruler still slicing through part of the ring or part of the middle. This ruler has, if I am using this correctly, no aura.")


@lesson.step("Four dots are enough to prove it", beat="hook")
def _():
    lesson.say("XOR makes the failure tiny enough to prove. It has four points, and opposite corners match.")
    guess = lesson.predict(
        "Can any straight line split these opposite-corner answers perfectly?",
        ["Yes", "No", "Only if the line is diagonal"],
        correct=1,
        why="A diagonal can scoop up one matching pair of corners. Then it scoops up the other matching pair too. The ruler has nowhere clean to land!",
        key="ch03_xor_line",
    )
    if guess is None:
        return

    fig, ax = lesson.figure(5, 4.5)
    scatter_2d(X_xor, y_xor, ax=ax, size=120)
    for i, (x1, x2) in enumerate(X_xor):
        ax.text(x1 + 0.03, x2 + 0.03, str(int(y_xor[i])), fontsize=12)
    ax.set_title("XOR: opposite corners match")
    lesson.show(fig)
    lesson.look_for("the diagonals: the two red points are not neighbors, and the two blue points are not neighbors either.")


@lesson.step("The contradiction", beat="byhand")
def _():
    lesson.say(
        """
Assume a perfect line exists. Its score is **w1·x1 + w2·x2 + b**. Red points need
positive scores; blue points need negative scores.
"""
    )
    st.markdown(
        """
| point | answer | what the line would need |
|---|---|---|
| (0, 0) | blue | b < 0 |
| (1, 1) | blue | w1 + w2 + b < 0 |
| (1, 0) | red | w1 + b > 0 |
| (0, 1) | red | w2 + b > 0 |
"""
    )
    lesson.say(
        """
Add the two red demands and you get **w1 + w2 + 2b > 0**. Add the two blue
demands and you get the same left side, but now **w1 + w2 + 2b < 0**.
"""
    )
    lesson.aha("The same number cannot be bigger than zero and smaller than zero. That is why no straight line can solve XOR.")
    lesson.jargon("linearly separable", "A dataset is linearly separable if one straight line can split it perfectly.")


@lesson.step("Invent a height", beat="seeit")
def _():
    guess = lesson.predict(
        "If we add x3 = x1² + x2² to circle data, what happens to points far from the middle?",
        ["They rise higher", "They sink lower", "Nothing changes"],
        correct=0,
        why="x1² + x2² is distance-from-the-middle squared. Ring points climb like beads on stilts; middle points stay low!",
        key="ch03_lift_predict",
    )
    if guess is None:
        return

    lesson.say(
        """
A circle problem is hard in **x1, x2** because "inside or ring?" is really about
distance from the middle. So we add **x3 = x1² + x2²**.
"""
    )
    lesson.mermaid(
        """
flowchart LR
    A[Original x1 and x2] --> B[Add x3 = x1^2 + x2^2]
    B --> C[Lift into 3D]
    C --> D[Cut with a flat plane]
    D --> E[Drop back to 2D]
    E --> F[Circle boundary]
""",
        height=260,
    )
    lesson.look_for("the escape route: make height from distance, cut flat, then look back down.")


@lesson.step("A flat cut becomes a circle", beat="seeit")
def _():
    X, y = toy_shape("circles", n=220, noise=0.08, seed=1)
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2
    colors = np.where(y == 1, WARM, COOL)
    fig3 = go.Figure(
        data=[go.Scatter3d(x=X[:, 0], y=X[:, 1], z=r2, mode="markers", marker=dict(size=4, color=colors))]
    )
    plane_x, plane_y = np.meshgrid(np.linspace(-1.8, 1.8, 2), np.linspace(-1.8, 1.8, 2))
    plane_z = np.full_like(plane_x, 0.55)
    fig3.add_trace(go.Surface(x=plane_x, y=plane_y, z=plane_z, opacity=0.35, showscale=False, colorscale=[[0, ACCENT], [1, ACCENT]]))
    fig3.update_layout(height=520, scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x1² + x2²"))
    st.plotly_chart(fig3, width="stretch")
    lesson.look_for("the cut is flat in the lifted picture, but its shadow on the floor is curved.")
    lesson.say("A flat slice at **x3 = 0.55** casts the circle **x1² + x2² = 0.55** below it.")


@lesson.step("XOR gets its own new feature", beat="seeit")
def _():
    lesson.say("XOR has its own escape hatch: add **x3 = x1 × x2**.")
    X3 = np.c_[X_xor, X_xor[:, 0] * X_xor[:, 1]]
    score = X3[:, 0] + X3[:, 1] - 2 * X3[:, 2] - 0.5
    st.dataframe(
        {"x1": X3[:, 0], "x2": X3[:, 1], "x1*x2": X3[:, 2], "new straight score": score, "answer": y_xor},
        hide_index=True,
    )
    lesson.look_for("the (1, 1) row. The product feature turns that corner into the special case.")
    lesson.say("For **(1, 1)** the score is **1 + 1 - 2(1) - 0.5 = -0.5**, blue. For **(1, 0)** it is **0.5**, red.")


@lesson.step("Models can bend for you", beat="play")
def _():
    lesson.say(
        """
Adding a feature is one way to bend the answer back in the original picture.
Another way is to use a model that builds bends itself.
"""
    )
    shape = st.selectbox("Shape", ["circles", "xor", "moons"], index=0, key="ch03_bendy_shape")
    Xb, yb = toy_shape(shape, n=240, noise=0.15, seed=3)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xb, yb)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=600, solver="lbfgs", random_state=1).fit(Xb, yb)
    cols = st.columns(2)
    with cols[0]:
        fig, ax = lesson.figure(5.3, 4.6)
        decision_boundary(lambda G: tree.predict(G), Xb, yb, ax=ax, shade_confidence=False, title="A decision tree bends by making boxes")
        lesson.show(fig)
    with cols[1]:
        fig, ax = lesson.figure(5.3, 4.6)
        decision_boundary(lambda G: mlp.predict_proba(G)[:, 1], Xb, yb, ax=ax, shade_confidence=True, title="A tiny neural net bends smoothly")
        lesson.show(fig)
    lesson.look_for("the two styles of bend: square corners on the left, a smoother curve on the right.")


@lesson.step("Polynomial features for real", beat="forreal")
def _():
    lesson.say(
        """
scikit-learn can add polynomial features for us, then fit a straight model in
that bigger feature space. Degree 1 means no extra bend. Higher degree adds more
terms, which gives the boundary more ways to curve.
"""
    )
    degree = st.slider("Polynomial degree", 1, 8, 2, key="ch03_degree")
    cols = st.columns(2)
    for col, real_shape in zip(cols, ["moons", "circles"]):
        Xm, ym = toy_shape(real_shape, n=240, noise=0.18, seed=8)
        pipe = make_pipeline(PolynomialFeatures(degree=degree), LogisticRegression(max_iter=1000)).fit(Xm, ym)
        with col:
            fig, ax = lesson.figure(5.3, 4.6)
            decision_boundary(lambda G, p=pipe: p.predict_proba(G)[:, 1], Xm, ym, ax=ax, shade_confidence=True, title=f"{real_shape}, degree {degree}")
            lesson.show(fig)
    lesson.look_for("when the curve starts chasing individual dots. That is over-studying, and it becomes a major problem later.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Rank the six toy shapes.** Which ones can one straight line handle?
2. **Prove XOR again.** Explain the contradiction without using the word equation.
3. **Invent a feature for stripes.** Hint: something that repeats as x1 moves.
"""
    )
    lesson.kid_corner(
        "If a rope cannot separate a donut from its hole on the floor, lift the donut pieces onto chairs. "
        "Now a flat tray can separate high from low."
    )


lesson.finish()
