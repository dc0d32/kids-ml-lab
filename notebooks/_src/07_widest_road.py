# %% [markdown]
# # Chapter 07 · The Widest Road
#
# ### Separate with the biggest gap.
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
# ## 🎣 Start here
#
# Chapter 2's perceptron stops when it finds **any** line that separates the dots. That is
# enough for yesterday's dots, but it may be a nervous choice for tomorrow's dot.
#
# Imagine a new point lands a tiny bit away from where you expected. A line that hugs one
# class has no shoulder; one small wiggle can shove the new point across the road.
#
# Your instinct says: pick the line with the biggest empty gap around it. Wider roads
# survive small surprises better!

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
# Notice the green road leaves room on both sides. The grey roads separate the training
# dots too, but one side is close enough that a small measurement wiggle could cross it.
#
# > 🧸 **Little Kid Corner** — Imagine walking between two puddles. You do not walk
# > touching one puddle. You take the widest dry path because your foot might wobble.

# %% [markdown]
# ## ✏️ Work it out
#
# The safety gap is the distance from the road to the closest dot on either side. A road
# is only as safe as its closest danger.
#
# For `x = 2.5`, the nearest blue dot has `x = 2`, so the blue gap is `2.5 - 2 = 0.5`.
# The nearest red dot has `x = 4`, so the red gap is `4 - 2.5 = 1.5`. The smallest gap is
# `0.5`.

# %%
pd.DataFrame({"x1": X_hand[:, 0], "x2": X_hand[:, 1], "class": ["blue"] * 3 + ["red"] * 3})

# %%
candidates

# %% [markdown]
# For `x = 3.0`, both nearest gaps are `1.0`, so its smallest gap is bigger. Both roads
# fit the old dots. The wider road is the one we trust more for dots we have not seen.
#
# > 📖 **Grown-ups call this:** a **support vector machine** chooses the separating road
# > with the widest safe gap.
#
# > 📖 **Grown-ups call this:** the **margin** is the empty road between the classes.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(7)

# %% [markdown]
# ## 👀 Take a look
#
# Once the road is as wide as possible, most dots are not pushing on it. They sit far
# back on their own side, so nudging them a little would not shrink the road.
#
# The closest dots are different. They touch the edge of the road like fence posts. Move
# or remove one of those, and the widest possible road may change.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, remove in zip(axes, ["none", "non-support", "support"]):
    model, X, y = fit_linear_svm(remove=remove)
    plot_linear_svm_margin(model, X, y, ax=ax, title=f"removed: {remove}")
plt.show()

# %% [markdown]
# The ringed points are the support vectors. Delete a far-away point and the road barely
# moves; that dot was, technically, an NPC. Delete a ringed point and the road can jump
# because the old road was resting against it.

# %% [markdown]
# ## 🎛️ Your turn
#
# Real data is messy, so the road sometimes has to choose: stay wide, or bend hard to fix
# every training dot.
#
# In these next pictures we use a distance road: each point tugs on nearby space, and the
# tug fades as you move away. Grown-ups call this RBF.
#
# `C` is the strictness knob. Low C says, "keep a wide road, even if a few training dots
# are on the wrong side." High C says, "training mistakes are expensive," so the road
# narrows or bends to chase them.
#
# `gamma` is the reach knob for the RBF road. Low gamma means each point reaches far,
# making broad smooth shapes. High gamma means each point reaches a short distance, which
# can create tiny islands.
#
# > 📖 **Grown-ups call this:** **RBF** is short for radial basis function: an SVM road
# > style where nearby points tug more than far-away points.

# %%
X, y, model = fit_svm_shape("circles", kernel="rbf", C=3.0, gamma=1.0, noise=0.18, seed=2)
fig, ax = plt.subplots(figsize=(6.2, 5))
decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False, title="RBF SVM")
plt.show()

# %% [markdown]
# Watch what happens near noisy dots. High C and high gamma can make the boundary curl
# around individual points. It looks impressive on training data and may crack on new
# data.
#
# > ⚠️ **Careful** A kid-repeat version: **C is strictness; gamma is reach**. Strict and
# > short-reach can memorise islands. Forgiving and long-reach gives a smoother road.
#
# One more name before the real-data picture: grown-ups call a road style a **kernel**.
# A linear kernel draws one straight road. A polynomial kernel makes smooth curves. An
# RBF kernel uses distance from points, so it can make smooth islands. Same SVM, different
# idea of what a road can be.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, kernel in zip(axes, ["linear", "poly", "rbf"]):
    X, y, model = fit_svm_shape("circles", kernel=kernel, C=3.0, gamma=1.0, noise=0.18, seed=2)
    decision_boundary(model.predict, X, y, ax=ax, steps=140, shade_confidence=False, title=f"{kernel} kernel")
plt.show()

# %% [markdown]
# Same data, three road styles. The word **kernel** belongs with the picture of the road
# changing shape.

# %% [markdown]
# ## 💻 In real code
#
# These are the same Palmer penguins from Chapter 04, measured in a different way. Chapter
# 04 used flipper length and weight; this road uses beak length and beak depth so the
# picture still has two axes.
#
# Each row is one penguin, and the target is species. The ringed penguins are the ones
# close enough to hold the road in place.

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
# Notice how many penguins are not ringed. They still helped show where the classes live,
# but they are not the points that set the final road width.
#
# Now connect back to Chapter 3. There, circles became easier after we invented a new
# `radius²` feature and lifted the data into a space where a straight slice could work.
# An RBF SVM uses the same kind of idea without asking you to build all those extra
# columns by hand.
#
# ```mermaid
# graph LR
#     A[2D circle dots] --> B[imagine extra features]
#     B --> C[straight cut there]
#     C --> D[curved road back here]
# ```
#
# The diagram is the kernel trick in kid language: lift the dots onto an easier cutting
# board, slice there, then read the cut back in the original picture.

# %%
X_c, y_c, lifted_predict = fit_circles_lifted()
X_rbf, y_rbf, rbf_model = fit_svm_shape("circles", kernel="rbf", C=2.0, gamma=1.0, noise=0.12, seed=2)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
decision_boundary(lifted_predict, X_c, y_c, ax=axes[0], steps=140, shade_confidence=False, title="linear SVM + radius feature")
decision_boundary(rbf_model.predict, X_rbf, y_rbf, ax=axes[1], steps=140, shade_confidence=False, title="RBF shortcut")
plt.show()

# %% [markdown]
# Look for the same lesson in both panels: a straight idea in a lifted space can become a
# curved boundary in the original space.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Add noise and find the C where the road starts chasing it.
# 2. Make gamma so large that the RBF SVM memorises islands.
# 3. On circles, beat the RBF SVM with a linear SVM plus your own `x1² + x2²` feature.
# 4. 🧸 **Little Kid Corner:** Put two sticker colours on a table. Draw the widest road between them.
#
# ---
# **Next up:** Chapter 08 · *The Model Zoo* — where we compare guessers without fooling
# ourselves.
