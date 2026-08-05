# %% [markdown]
# # Chapter 03 · When a Ruler Isn't Enough
#
# ### Some things a straight line cannot do.
#
# *Part 1 · Classical models*
#
# ---
#
# This notebook is the same chapter as the app, but with the code showing.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

from kidsml.datasets import toy_shape, xor_exact
from kidsml.linear import predict_side
from kidsml.plots import ACCENT, COOL, WARM, decision_boundary, draw_line, scatter_2d, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Chapter 2 gave us a ruler: one straight line can choose red or blue. Now the line
# runs out of road.
#
# Try the ruler on circles. The middle wants one answer and the ring wants the
# other. A single line can cut left from right, or top from bottom, but it cannot
# wrap around the middle. The ruler is stuck being a ruler, and I believe this means it has no aura!

# %%
X_fail, y_fail = toy_shape("circles", n=160, noise=0.08, seed=0)
w1_fail, w2_fail, b_fail = 1.0, 0.0, 0.0
mistakes = int((predict_side(X_fail, w1_fail, w2_fail, b_fail) != y_fail).sum())
print("circle mistakes with one line:", mistakes)

fig, ax = plt.subplots(figsize=(6, 4.6))
decision_boundary(lambda G: predict_side(G, w1_fail, w2_fail, b_fail), X_fail, y_fail, ax=ax, shade_confidence=False)
draw_line(w1_fail, w2_fail, b_fail, ax=ax)
ax.set_title("Try to make circles perfect with one line")
plt.show()

# %% [markdown]
# Notice the best-looking ruler still slices through part of the ring or part of the middle.
#
# XOR makes the failure tiny enough to prove. It has four points. Opposite corners
# match. If one line cannot solve four dots, then the problem is the shape of the
# boundary, not the amount of data.

# %%
X_xor, y_xor = xor_exact()
fig, ax = plt.subplots(figsize=(5, 4.5))
scatter_2d(X_xor, y_xor, ax=ax, size=120)
for i, (x1, x2) in enumerate(X_xor):
    ax.text(x1 + 0.03, x2 + 0.03, str(int(y_xor[i])), fontsize=12)
ax.set_title("XOR: opposite corners match")
plt.show()

# %% [markdown]
# Look diagonally: the two red points are not neighbors, and the two blue points are
# not neighbors either.

# %% [markdown]
# ## ✏️ Work it out
#
# Assume a perfect line exists. Its score is **w1·x1 + w2·x2 + b**. Red points need
# positive scores; blue points need negative scores.
#
# Here is what each XOR corner demands:
#
# | point | answer | what the line would need |
# |---|---|---|
# | (0, 0) | blue | b < 0 |
# | (1, 1) | blue | w1 + w2 + b < 0 |
# | (1, 0) | red | w1 + b > 0 |
# | (0, 1) | red | w2 + b > 0 |
#
# Now add the two red demands. From **w1 + b > 0** and **w2 + b > 0**, the left
# sides add to **w1 + w2 + 2b**, and two positive things add to something positive:
# **w1 + w2 + 2b > 0**.
#
# Add the two blue demands. From **b < 0** and **w1 + w2 + b < 0**, the same left
# side appears, but now it must be negative: **w1 + w2 + 2b < 0**.
#
# The same number cannot be bigger than zero and smaller than zero. That is why no
# straight line can solve XOR.
#
# > 📖 **Grown-ups call this:** **linearly separable** — one straight line can split the
# > data perfectly.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(3)

# %% [markdown]
# ## 👀 Take a look
#
# A circle problem is hard in **x1, x2** because "inside or ring?" is really about
# distance from the middle. So we add a new feature:
#
# **x3 = x1² + x2²**
#
# Point **(2, 0)** becomes **x3 = 2² + 0² = 4**. Point **(0.3, 0.4)** becomes
# **0.3² + 0.4² = 0.09 + 0.16 = 0.25**. The ring rises; the middle stays low. Pop!
#
# ```mermaid
# flowchart LR
#     A[Original x1 and x2] --> B[Add x3 = x1^2 + x2^2]
#     B --> C[Lift into 3D]
#     C --> D[Cut with a flat plane]
#     D --> E[Drop back to 2D]
#     E --> F[Circle boundary]
# ```
#
# The diagram says the trick: make height from distance, cut flat, then look back down. A curve appears on the floor!

# %%
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
fig3.show()

# %% [markdown]
# Notice that the cut is flat in the lifted picture. The boundary back on the
# floor is curved because the height came from **x1² + x2²**. A flat slice at
# **x3 = 0.55** casts the circle **x1² + x2² = 0.55** below it.

# %%
X3 = np.c_[X_xor, X_xor[:, 0] * X_xor[:, 1]]
score = X3[:, 0] + X3[:, 1] - 2 * X3[:, 2] - 0.5
pd.DataFrame({"x1": X3[:, 0], "x2": X3[:, 1], "x1*x2": X3[:, 2], "new straight score": score, "answer": y_xor})

# %% [markdown]
# XOR has its own escape hatch: add **x3 = x1 × x2**. For **(1, 1)** the score is
# **1 + 1 - 2(1) - 0.5 = -0.5**, blue. For **(1, 0)** it is
# **1 + 0 - 2(0) - 0.5 = 0.5**, red.

# %% [markdown]
# ## 🎛️ Your turn
#
# Adding a feature is one way to bend the answer back in the original picture.
# Another way is to use a model that builds bends itself.
#
# A decision tree bends with boxy cuts. A tiny neural net bends smoothly. Both are
# still making regions of red and blue; they are no longer trapped with one ruler.

# %%
Xb, yb = toy_shape("circles", n=240, noise=0.15, seed=3)
tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xb, yb)
mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=600, solver="lbfgs", random_state=1).fit(Xb, yb)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
decision_boundary(lambda G: tree.predict(G), Xb, yb, ax=axes[0], shade_confidence=False, title="Decision tree")
decision_boundary(lambda G: mlp.predict_proba(G)[:, 1], Xb, yb, ax=axes[1], shade_confidence=True, title="Tiny neural net")
plt.show()

# %% [markdown]
# Look at the two styles of bend: square corners on the left, a smoother curve on the right.

# %% [markdown]
# ## 💻 In real code
#
# scikit-learn can add polynomial features for us, then fit a straight model in
# that bigger feature space. Degree 1 means no extra bend. Higher degree adds more
# terms, which gives the boundary more ways to curve.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for row, shape in enumerate(["moons", "circles"]):
    Xm, ym = toy_shape(shape, n=220, noise=0.18, seed=8)
    for ax, degree in zip(axes[row], [1, 3, 8]):
        pipe = make_pipeline(PolynomialFeatures(degree=degree), LogisticRegression(max_iter=1000)).fit(Xm, ym)
        decision_boundary(lambda G, p=pipe: p.predict_proba(G)[:, 1], Xm, ym, ax=ax, shade_confidence=True, title=f"{shape}, degree {degree}")
plt.show()

# %% [markdown]
# Degree 1 is straight. Degree 3 bends. Degree 8 may get wild and chase individual
# dots. That wild end is the seed of overfitting: over-studying the training dots.

# %% [markdown]
# ## 🏆 Go further
#
# 1. **Rank the six toy shapes.** Which ones can one straight line handle?
# 2. **Prove XOR again.** Explain the contradiction without using equations.
# 3. **Invent a feature for stripes.** Hint: something that repeats as x1 moves.
# 4. 🧸 **Little Kid Corner:** If a rope cannot separate a donut from its hole on the
#    floor, lift the donut pieces onto chairs. Now a flat tray can separate high from low.
#
# ---
# **Next up:** Chapter 04 · *Maybe, Probably, Definitely* — where a model learns to say
# "I am not sure."
