"""Chapter 02 · Lines That Decide."""

from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.linear_model import Perceptron

from kidsml import ui
from kidsml.datasets import toy_shape, two_blobs_tiny
from kidsml.linear import mistake_count, perceptron_history, predict_side, score_line
from kidsml.nn_numpy import perceptron_step
from kidsml.plots import ACCENT, MUTED, decision_boundary, draw_line, scatter_2d

ui.page_setup(2)

# ---------------------------------------------------------------------------
ui.beat("hook")
st.markdown(
    """
Chapter 1's line answered **how much?**

Now the line answers **which one?** One side is blue. The other side is red. A single
line can split the whole world in two.
"""
)
ui.jargon("perceptron", "An old-school model that decides which side of a line a point is on.")

X_tiny, y_tiny = two_blobs_tiny()

# ---------------------------------------------------------------------------
ui.beat("byhand")
w_start = np.array([1.0, 1.0])
b_start = -8.0
scores = score_line(X_tiny[:5], w_start[0], w_start[1], b_start)
hand = {
    "x1": X_tiny[:5, 0],
    "x2": X_tiny[:5, 1],
    "score = x1 + x2 - 8": scores,
    "guess": np.where(scores > 0, "red", "blue"),
    "truth": np.where(y_tiny[:5] == 1, "red", "blue"),
}
st.dataframe(hand, hide_index=True, use_container_width=True)
st.markdown(
    "Point (6, 5) is red. Our score is **6 + 5 - 8 = 3**, so we guess red. "
    "Good. Now try a line that gets it wrong: w = (1, 1), b = -20."
)
w_bad = np.array([1.0, 1.0])
b_bad = -20.0
w_after, b_after, was_wrong = perceptron_step(w_bad, b_bad, X_tiny[5], y_tiny[5])
st.write(f"Wrong? **{was_wrong}**. New w = **({w_after[0]:.0f}, {w_after[1]:.0f})**, new b = **{b_after:.0f}**.")
st.markdown("You added the red point's coordinates to the weights. You trained a model by hand.")

# ---------------------------------------------------------------------------
ui.beat("seeit")
fig, axes = ui.two_figures(5.0, 4.4)
for ax, w_now, b_now, title in [
    (axes[0], w_bad, b_bad, "Before the one update"),
    (axes[1], w_after, b_after, "After the one update"),
]:
    scatter_2d(X_tiny, y_tiny, ax=ax)
    ax.scatter([X_tiny[5, 0]], [X_tiny[5, 1]], s=180, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
    draw_line(w_now[0], w_now[1], b_now, ax=ax)
    ax.set_title(title)
ui.show(fig)

hist = perceptron_history(X_tiny, y_tiny, w=(0, 0), b=0, steps=12)
step = st.slider("Perceptron learning step", 0, len(hist) - 1, min(4, len(hist) - 1))
row = hist.iloc[step]
fig, ax = ui.figure(6, 4.6)
decision_boundary(lambda G: predict_side(G, row.w1, row.w2, row.b), X_tiny, y_tiny, ax=ax, shade_confidence=False)
draw_line(row.w1, row.w2, row.b, ax=ax)
ax.set_title(f"Step {step}: {int(row.mistakes)} mistake(s)")
ui.show(fig)

# ---------------------------------------------------------------------------
ui.beat("play")
X, y = toy_shape("blobs", n=180, noise=0.25, seed=2)
col_a, col_b = st.columns([1, 2], gap="large")
with col_a:
    w1 = st.slider("w1", -5.0, 5.0, 1.0, 0.1)
    w2 = st.slider("w2", -5.0, 5.0, 1.0, 0.1)
    b = st.slider("b", -3.0, 3.0, 0.0, 0.1)
    st.metric("Mistakes", mistake_count(X, y, w1, w2, b))
with col_b:
    fig, ax = ui.figure(6, 5)
    decision_boundary(lambda G: predict_side(G, w1, w2, b), X, y, ax=ax, shade_confidence=False)
    draw_line(w1, w2, b, ax=ax)
    ax.arrow(0, 0, w1 * 0.25, w2 * 0.25, color=ACCENT, width=0.025, length_includes_head=True)
    ax.text(w1 * 0.28, w2 * 0.28, "w arrow", color=ACCENT)
    ax.set_title("w points toward red. b slides the line.")
    ui.show(fig)

# ---------------------------------------------------------------------------
ui.beat("forreal")
model = Perceptron(max_iter=1000, random_state=0).fit(X, y)
st.code("Perceptron(max_iter=1000).fit(X, y)", language="python")
st.write(f"On clean blobs, scikit-learn gets **{model.score(X, y):.0%}** right.")
cols = st.columns(2)
for col, shape in zip(cols, ["moons", "circles"]):
    Xr, yr = toy_shape(shape, n=220, noise=0.18, seed=4)
    m = Perceptron(max_iter=1000, random_state=0).fit(Xr, yr)
    with col:
        fig, ax = ui.figure(5.2, 4.5)
        decision_boundary(lambda G, mm=m: mm.predict(G), Xr, yr, ax=ax, shade_confidence=False, title=f"{shape}: {m.score(Xr, yr):.0%} right")
        ui.show(fig)
ui.careful("Circles need a boundary that wraps around. One line cannot do that.")

# ---------------------------------------------------------------------------
ui.beat("challenge")
st.markdown(
    """
1. **Beat the algorithm.** Use the sliders until the mistake counter reaches zero.
2. **Set b to 0.** What can the line no longer do? Hint: it must pass through (0, 0).
3. **Make the blobs overlap.** A perceptron only settles when perfection is possible.
4. 🧸 **Little Kid Corner:** Lay a pencil between two piles of toys. One side is red team,
   the other is blue team. Move the pencil until nobody is on the wrong side.
"""
)
ui.worksheet_link(2)
