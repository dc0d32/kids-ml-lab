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

from kidsml import ui
from kidsml.datasets import toy_shape, xor_exact
from kidsml.linear import predict_side
from kidsml.plots import ACCENT, COOL, WARM, decision_boundary, draw_line, scatter_2d

ui.page_setup(3)

# ---------------------------------------------------------------------------
ui.beat("hook")
st.markdown(
    """
Here are Chapter 2's sliders on circles. Try every ruler line you want. The middle and
the ring refuse to separate.

Now make the problem even smaller: XOR has four points and four answers. No line on
Earth works.
"""
)
X_xor, y_xor = xor_exact()
X_fail, y_fail = toy_shape("circles", n=160, noise=0.08, seed=0)
col_a, col_b = st.columns([1, 2], gap="large")
with col_a:
    w1_fail = st.slider("circle w1", -5.0, 5.0, 1.0, 0.2)
    w2_fail = st.slider("circle w2", -5.0, 5.0, 0.0, 0.2)
    b_fail = st.slider("circle b", -3.0, 3.0, 0.0, 0.2)
    mistakes = int((predict_side(X_fail, w1_fail, w2_fail, b_fail) != y_fail).sum())
    st.metric("Circle mistakes", mistakes)
with col_b:
    fig, ax = ui.figure(6, 4.6)
    decision_boundary(lambda G: predict_side(G, w1_fail, w2_fail, b_fail), X_fail, y_fail, ax=ax, shade_confidence=False)
    draw_line(w1_fail, w2_fail, b_fail, ax=ax)
    ax.set_title("Try to make circles perfect with one line")
    ui.show(fig)

fig, ax = ui.figure(5, 4.5)
scatter_2d(X_xor, y_xor, ax=ax, size=120)
for i, (x1, x2) in enumerate(X_xor):
    ax.text(x1 + 0.03, x2 + 0.03, str(int(y_xor[i])), fontsize=12)
ax.set_title("XOR: opposite corners match")
ui.show(fig)

# ---------------------------------------------------------------------------
ui.beat("byhand", "A tiny proof with inequalities.")
st.markdown(
    """
For XOR, red means the score must be positive. Blue means negative.

| point | answer | what the line would need |
|---|---|---|
| (0, 0) | blue | b < 0 |
| (1, 1) | blue | w1 + w2 + b < 0 |
| (1, 0) | red | w1 + b > 0 |
| (0, 1) | red | w2 + b > 0 |

Add the two red rows: **w1 + w2 + 2b > 0**.

Add the two blue rows: **w1 + w2 + 2b < 0**.

The same number cannot be bigger than zero and smaller than zero. The ruler loses.
"""
)
ui.jargon("linearly separable", "A dataset is linearly separable if one straight line can split it perfectly.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Escape route 1: invent a better feature.")
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
st.plotly_chart(fig3, use_container_width=True)
st.markdown("We did not get a bendy model. We got a straight model in a cleverer space. From above, that flat slice looks like a circle.")

X3 = np.c_[X_xor, X_xor[:, 0] * X_xor[:, 1]]
score = X3[:, 0] + X3[:, 1] - 2 * X3[:, 2] - 0.5
st.dataframe(
    {"x1": X3[:, 0], "x2": X3[:, 1], "x1*x2": X3[:, 2], "new straight score": score, "answer": y_xor},
    hide_index=True,
)
st.markdown("XOR has its own escape hatch: add **x3 = x1 × x2**. The score is positive exactly on the red rows.")

# ---------------------------------------------------------------------------
ui.beat("play", "Escape route 2: let the boundary bend.")
shape = st.selectbox("Shape", ["circles", "xor", "moons"], index=0)
Xb, yb = toy_shape(shape, n=240, noise=0.15, seed=3)
tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xb, yb)
mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=600, solver="lbfgs", random_state=1).fit(Xb, yb)
cols = st.columns(2)
with cols[0]:
    fig, ax = ui.figure(5.3, 4.6)
    decision_boundary(lambda G: tree.predict(G), Xb, yb, ax=ax, shade_confidence=False, title="A decision tree bends by making boxes")
    ui.show(fig)
with cols[1]:
    fig, ax = ui.figure(5.3, 4.6)
    decision_boundary(lambda G: mlp.predict_proba(G)[:, 1], Xb, yb, ax=ax, shade_confidence=True, title="A tiny neural net bends smoothly")
    ui.show(fig)
st.markdown("The next five chapters are five different ways of getting that bend.")

# ---------------------------------------------------------------------------
ui.beat("forreal")
degree = st.slider("Polynomial degree", 1, 8, 2)
cols = st.columns(2)
for col, real_shape in zip(cols, ["moons", "circles"]):
    Xm, ym = toy_shape(real_shape, n=240, noise=0.18, seed=8)
    pipe = make_pipeline(PolynomialFeatures(degree=degree), LogisticRegression(max_iter=1000)).fit(Xm, ym)
    with col:
        fig, ax = ui.figure(5.3, 4.6)
        decision_boundary(lambda G, p=pipe: p.predict_proba(G)[:, 1], Xm, ym, ax=ax, shade_confidence=True, title=f"{real_shape}, degree {degree}")
        ui.show(fig)
ui.careful("High degree can wiggle so much that it starts memorising dots. That is over-studying.")

# ---------------------------------------------------------------------------
ui.beat("challenge")
st.markdown(
    """
1. **Rank the six toy shapes.** Which ones can one straight line handle?
2. **Prove XOR again.** Explain the contradiction without using the word equation.
3. **Invent a feature for stripes.** Hint: something that repeats as x1 moves.
4. 🧸 **Little Kid Corner:** If a rope cannot separate a donut from its hole on the floor,
   lift the donut pieces onto chairs. Now a flat tray can separate high from low.
"""
)
ui.worksheet_link(3)
