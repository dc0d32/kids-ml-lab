# %% [markdown]
# # Chapter 07 · The Widest Road
#
# ### Don't just separate — separate with the biggest gap.
#
# *Part 1 · Classical models*
#
# ---
#
# Chapter 3 showed shapes a straight line cannot split. Chapter 7 adds another bendy
# escape: a road that can stay straight, curve, or make islands.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from kidsml.plots import decision_boundary, scatter_2d, use_house_style
from kidsml.trees import (
    fit_circles_lifted,
    fit_linear_svm,
    fit_svm_shape,
    penguin_svm,
    plot_linear_svm_margin,
    svm_hand_points,
)

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# Chapter 2's perceptron stops when it finds **any** line that separates the dots. But
# several perfect lines can exist. Which one would you trust for a new point?
#
# Your instinct says: pick the line with the biggest empty gap around it.
#
# > 🧸 **Little Kid Corner** — Imagine walking between two puddles. You do not walk
# > touching one puddle. You take the widest dry path.

# %%
X_hand, y_hand, candidates = svm_hand_points()
fig, ax = plt.subplots(figsize=(6, 4.5))
scatter_2d(X_hand, y_hand, ax=ax)
ax.axvline(2.25, color="#94A3B8", linewidth=2, label="hugs blue")
ax.axvline(3.75, color="#94A3B8", linewidth=2, linestyle="--", label="hugs red")
ax.axvline(3.0, color="#10B981", linewidth=3, label="middle road")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Measure the safety gap. The road with the biggest smallest gap wins.

# %%
pd.DataFrame({"x1": X_hand[:, 0], "x2": X_hand[:, 1], "class": ["blue"] * 3 + ["red"] * 3})

# %%
candidates

# %% [markdown]
# > 📖 **Grown-ups call this:** a **support vector machine** chooses the separating road
# > with the widest safe gap.
#
# > 📖 **Grown-ups call this:** the **margin** is the empty road between the classes.

# %% [markdown]
# ## 👀 See It
#
# The ringed points hold the road in place. Remove a far-away point and the road barely
# moves. Remove a ringed point and the road jumps.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, remove in zip(axes, ["none", "non-support", "support"]):
    model, X, y = fit_linear_svm(remove=remove)
    plot_linear_svm_margin(model, X, y, ax=ax, title=f"removed: {remove}")
plt.show()

# %% [markdown]
# ## 🎛️ Play With It
#
# `C` asks how much we care about getting every training point right versus keeping the
# road wide. `gamma` asks how far each point's influence reaches for the RBF road.

# %%
X, y, model = fit_svm_shape("circles", kernel="rbf", C=3.0, gamma=1.0, noise=0.18, seed=2)
fig, ax = plt.subplots(figsize=(6.2, 5))
decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False, title="RBF SVM")
plt.show()

# %% [markdown]
# > ⚠️ **Careful** Low C keeps the road wide and forgives a few mistakes. High C narrows
# > the road to chase every dot. Huge gamma can draw a tiny island around each point.

# %% [markdown]
# ## 💻 For Real
#
# Penguins are real data. We use two beak measurements so the road is plottable.

# %%
X_peng, y_peng, species, peng_model, support = penguin_svm()
fig, ax = plt.subplots(figsize=(6, 4.8))
decision_boundary(peng_model.predict, X_peng, y_peng, ax=ax, steps=150, shade_confidence=False, title="penguin species")
ax.scatter(support[:, 0], support[:, 1], s=95, facecolors="none", edgecolors="black", linewidths=1.4)
ax.set_xlabel("beak length (mm)")
ax.set_ylabel("beak depth (mm)")
plt.show()
print(f"support vectors: {len(support)} out of {len(X_peng)} penguins")

# %% [markdown]
# Back to Chapter 3: an RBF SVM is like the lifting-into-3D trick for circles, but it uses
# a shortcut so you do not build the extra columns by hand. The proof comes later.

# %%
X_c, y_c, lifted_predict = fit_circles_lifted()
X_rbf, y_rbf, rbf_model = fit_svm_shape("circles", kernel="rbf", C=2.0, gamma=1.0, noise=0.12, seed=2)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
decision_boundary(lifted_predict, X_c, y_c, ax=axes[0], steps=140, shade_confidence=False, title="linear SVM + radius feature")
decision_boundary(rbf_model.predict, X_rbf, y_rbf, ax=axes[1], steps=140, shade_confidence=False, title="RBF shortcut")
plt.show()

# %% [markdown]
# ## 🏆 Challenge
#
# 1. Add noise and find the C where the road starts chasing it.
# 2. Make gamma so large that the RBF SVM memorises islands.
# 3. On circles, beat the RBF SVM with a linear SVM plus your own `x1² + x2²` feature.
# 4. 🧸 **Little Kid Corner:** Put two sticker colours on paper. Draw the widest road between them.
#
# ---
# **Next up:** Chapter 08 · *The Model Zoo* — where we compare guessers without fooling
# ourselves.
