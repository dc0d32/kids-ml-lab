"""Chapter 07 · The Widest Road."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.plots import decision_boundary, draw_line, scatter_2d
from kidsml.trees import (
    fit_circles_lifted,
    fit_linear_svm,
    fit_svm_shape,
    penguin_svm,
    plot_linear_svm_margin,
    svm_hand_points,
)

ui.page_setup(7)

# ---------------------------------------------------------------------------
ui.beat("hook", "A perfect line can still be a nervous line.")

st.markdown(
    """
Chapter 2's perceptron stops when it finds **any** line that separates the dots.
But look at three perfect lines. Which one would you trust for a new point?

Your instinct says: pick the line with the biggest empty gap around it. That instinct has
a name.
"""
)

X_hand, y_hand, candidates = svm_hand_points()
fig, ax = ui.figure(6, 4.5)
scatter_2d(X_hand, y_hand, ax=ax)
ax.axvline(2.25, color="#94A3B8", linewidth=2, label="hugs blue")
ax.axvline(3.75, color="#94A3B8", linewidth=2, linestyle="--", label="hugs red")
ax.axvline(3.0, color="#10B981", linewidth=3, label="middle road")
ax.legend(fontsize=8)
ui.show(fig)

ui.little_kid_corner(
    "Imagine walking between two puddles. You do not walk touching one puddle. You take the widest dry path."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Measure the safety gap.")

st.dataframe(pd.DataFrame({"x1": X_hand[:, 0], "x2": X_hand[:, 1], "class": ["blue"] * 3 + ["red"] * 3}), hide_index=True)
st.dataframe(candidates, hide_index=True, use_container_width=True)
st.success("Both roads separate the dots. The one with the bigger smallest gap wins.")

ui.jargon("support vector machine", "A model that chooses the separating road with the widest safe gap.")
ui.jargon("margin", "The empty road between the two classes.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Only a few points hold the road in place.")

remove = st.selectbox("Remove a point", ["none", "non-support", "support"], key="svm_remove")
model, X, y = fit_linear_svm(remove=remove)
fig, ax = ui.figure(6, 4.8)
plot_linear_svm_margin(model, X, y, ax=ax, title=f"removed: {remove}")
ui.show(fig)

st.markdown(
    "Delete a point far away from the road and nothing moves. Delete a ringed point and "
    "the road jumps. The model politely ignores most of the crowd."
)

# ---------------------------------------------------------------------------
ui.beat("play", "How strict should the road be?")

shape = ui.shape_picker(default="circles", key="ch07_shape", include=("blobs", "moons", "circles", "xor"))
noise = ui.noise_slider(default=0.18, key="ch07_noise")
seed = ui.seed_slider(default=2, key="ch07_seed")
kernel = st.selectbox("Road style", ["linear", "poly", "rbf"], index=2, key="svm_kernel")
C = st.slider("C: care about every training dot", 0.1, 30.0, 3.0, 0.1, key="svm_C")
gamma = st.slider("gamma: how far each dot reaches", 0.05, 8.0, 1.0, 0.05, key="svm_gamma")

X, y, model = fit_svm_shape(shape, kernel=kernel, C=C, gamma=gamma, noise=noise, seed=seed)
fig, ax = ui.figure(6.2, 5)
decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False, title=f"{kernel} SVM")
ui.show(fig)

ui.careful(
    "Low C keeps the road wide and forgives a few mistakes. High C narrows the road to chase every dot. "
    "Huge gamma can draw a tiny island around each point."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "Penguins, with support vectors ringed.")

X_peng, y_peng, species, peng_model, support = penguin_svm()
fig, ax = ui.figure(6, 4.8)
decision_boundary(peng_model.predict, X_peng, y_peng, ax=ax, steps=150, shade_confidence=False, title="penguin species from two beak measurements")
ax.scatter(support[:, 0], support[:, 1], s=95, facecolors="none", edgecolors="black", linewidths=1.4)
ax.set_xlabel("beak length (mm)")
ax.set_ylabel("beak depth (mm)")
ui.show(fig)
st.info(f"This model depends on {len(support)} support vectors out of {len(X_peng)} penguins.")

st.markdown(
    "Back to Chapter 3: an RBF SVM is like the lifting-into-3D trick for circles, but it uses "
    "a shortcut so you do not build the extra columns by hand. The proof comes later."
)

X_c, y_c, lifted_predict = fit_circles_lifted()
X_rbf, y_rbf, rbf_model = fit_svm_shape("circles", kernel="rbf", C=2.0, gamma=1.0, noise=0.12, seed=2)
fig, axes = ui.two_figures(4.8, 4.2)
decision_boundary(lifted_predict, X_c, y_c, ax=axes[0], steps=140, shade_confidence=False, title="linear SVM + invented radius feature")
decision_boundary(rbf_model.predict, X_rbf, y_rbf, ax=axes[1], steps=140, shade_confidence=False, title="RBF SVM shortcut")
ui.show(fig)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Add noise and find the C where the road starts chasing it.
2. Make gamma so large that the RBF SVM memorises islands.
3. On circles, beat the RBF SVM with a linear SVM plus your own `x1² + x2²` feature.
4. 🧸 **Little Kid Corner:** Put two sticker colours on a table. Draw the widest road between them.
"""
)

ui.worksheet_link(7)
