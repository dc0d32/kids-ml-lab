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
Chapter 2's perceptron stops when it finds **any** line that separates the
dots. That is enough for yesterday's dots, but it may be a nervous choice for
tomorrow's dot.

Imagine a new point lands a tiny bit away from where you expected. A line that
hugs one class has no safety space; a small wiggle can push the new point to
the wrong side.

Your instinct says: pick the line with the biggest empty gap around it. Wider
roads survive small surprises better.
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

st.markdown(
    """
Notice the green road leaves room on both sides. The grey roads separate the
training dots too, but one side is close enough that a small measurement wiggle
could cross it.
"""
)

ui.little_kid_corner(
    "Imagine walking between two puddles. You do not walk touching one puddle. "
    "You take the widest dry path because your foot might wobble."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Measure the safety gap.")

st.markdown(
    """
The safety gap is the distance from the road to the closest dot on either side.
A road is only as safe as its closest danger.

For `x = 2.5`, the nearest blue dot has `x = 2`, so the blue gap is `2.5 - 2 =
0.5`. The nearest red dot has `x = 4`, so the red gap is `4 - 2.5 = 1.5`. The
smallest gap is `0.5`.
"""
)

st.dataframe(pd.DataFrame({"x1": X_hand[:, 0], "x2": X_hand[:, 1], "class": ["blue"] * 3 + ["red"] * 3}), hide_index=True)
st.dataframe(candidates, hide_index=True, use_container_width=True)

st.markdown(
    """
For `x = 3.0`, both nearest gaps are `1.0`, so its smallest gap is bigger.
Both roads fit the old dots. The wider road is the one we trust more for dots
we have not seen.
"""
)

st.success("Both roads separate the dots. The one with the bigger smallest gap wins.")

ui.jargon("support vector machine", "A model that chooses the separating road with the widest safe gap.")
ui.jargon("margin", "The empty road between the two classes.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Only a few points hold the road in place.")

st.markdown(
    """
Once the road is as wide as possible, most dots are not pushing on it. They are
far back inside their own side, so moving them a little would not shrink the
road.

The closest dots are different. They touch the edge of the road like fence
posts. Move or remove one of those, and the widest possible road may change.
"""
)

remove = st.selectbox("Remove a point", ["none", "non-support", "support"], key="svm_remove")
model, X, y = fit_linear_svm(remove=remove)
fig, ax = ui.figure(6, 4.8)
plot_linear_svm_margin(model, X, y, ax=ax, title=f"removed: {remove}")
ui.show(fig)

st.markdown(
    """
The ringed points are the support vectors. Delete a far-away point and the road
barely moves. Delete a ringed point and the road can jump because the old road
was resting against it.
"""
)

# ---------------------------------------------------------------------------
ui.beat("play", "How strict should the road be?")

st.markdown(
    """
Real data is messy, so the road sometimes has to choose: stay wide, or bend
hard to fix every training dot.

`C` is the strictness knob. Low C says, "keep a wide road, even if a few
training dots are on the wrong side." High C says, "training mistakes are
expensive," so the road narrows or bends to chase them.

`gamma` is the reach knob for the RBF road. Low gamma means each point reaches
far, making broad smooth shapes. High gamma means each point reaches a short
distance, which can create tiny islands.
"""
)

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

st.markdown(
    """
Watch what happens near noisy dots. High C and high gamma can make the boundary
curl around individual points, which feels impressive on training data and may
be fragile on new data.
"""
)

ui.careful(
    "A kid-repeat version: **C is strictness; gamma is reach**. Strict and short-reach "
    "can memorise islands. Forgiving and long-reach gives a smoother road."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "Penguins, with support vectors ringed.")

st.markdown(
    """
Penguins are real data, not toy dots. We use two beak measurements so the road
can be drawn. The ringed penguins are the ones close enough to hold the road in
place.
"""
)

X_peng, y_peng, species, peng_model, support = penguin_svm()
fig, ax = ui.figure(6, 4.8)
decision_boundary(peng_model.predict, X_peng, y_peng, ax=ax, steps=150, shade_confidence=False, title="penguin species from two beak measurements")
ax.scatter(support[:, 0], support[:, 1], s=95, facecolors="none", edgecolors="black", linewidths=1.4)
ax.set_xlabel("beak length (mm)")
ax.set_ylabel("beak depth (mm)")
ui.show(fig)
st.info(f"This model depends on {len(support)} support vectors out of {len(X_peng)} penguins.")

st.markdown(
    """
Notice how many penguins are not ringed. They still helped show where the
classes live, but they are not the points that set the final road width.
"""
)

st.markdown(
    """
Now connect back to Chapter 3. There, circles became easier after we invented a
new `radius²` feature and lifted the data into a space where a straight slice
could work. An RBF SVM uses the same kind of idea without asking you to build
all those extra columns by hand.
"""
)

ui.mermaid(
    """
graph LR
    A[2D circle dots] --> B[imagine extra features]
    B --> C[straight cut there]
    C --> D[curved road back here]
""",
    height=220,
)

st.markdown(
    """
The diagram is the kernel trick in kid language: make the data easier to cut
somewhere else, then read the cut back in the original picture.
"""
)

X_c, y_c, lifted_predict = fit_circles_lifted()
X_rbf, y_rbf, rbf_model = fit_svm_shape("circles", kernel="rbf", C=2.0, gamma=1.0, noise=0.12, seed=2)
fig, axes = ui.two_figures(4.8, 4.2)
decision_boundary(lifted_predict, X_c, y_c, ax=axes[0], steps=140, shade_confidence=False, title="linear SVM + invented radius feature")
decision_boundary(rbf_model.predict, X_rbf, y_rbf, ax=axes[1], steps=140, shade_confidence=False, title="RBF SVM shortcut")
ui.show(fig)

st.markdown(
    """
Look for the same lesson in both panels: a straight idea in a lifted space can
become a curved boundary in the original space.
"""
)

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
