# %% [markdown]
# # Chapter 21 · Squishing Dimensions
#
# ### PCA is picking the best shadow to cast.
#
# *Part 5 · Without answers*
#
# ---
#
# Hold up your hand and cast a shadow on a wall. Turn it. Some shadows shout hand. Other
# shadows become a pancake blob.
#
# Choosing the angle is the whole of PCA.
#
# The goal is not to keep every fact. Squishing always throws something away. The goal is
# to toss the quiet directions first and keep the directions where points still look
# different from each other.

# %%
import matplotlib.pyplot as plt

from kidsml import workbook
from kidsml.plots import use_house_style
from kidsml.unsupervised import (
    make_shadow_cloud,
    pca_hand_table,
    pca_shadow_answer,
    penguin_pca_table,
    plot_digits_pca,
    plot_digits_tsne,
    plot_eigendigits,
    plot_pca_linear_failure,
    plot_reconstruction,
    plot_shadow_2d,
    plot_variance_curve,
    shadow_projection,
    spread,
    variance_captured,
)

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Same hand. Same wall. Different angle. The useful shadow keeps the story.
#
# ```mermaid
# graph LR
#     A[High-dimensional cloud] --> B[Choose an angle]
#     B --> C[Cast a shadow]
#     C --> D[2D plot]
#     D --> E[Check what stayed spread out]
# ```
#
# Look at the last box. Spread means the points stayed far apart after the squish.
#
# > 🧸 **Little Kid Corner** — Use a flashlight and your hand. Turn your hand until the
# > shadow gives the best clue.

# %% [markdown]
# ## ✏️ Work it out
#
# Four points lie nearly sideways. Project onto x, then onto y. Spread is the sum of
# squared distances from the middle. Grown-ups call that measured spread **variance**.
#
# "Keep the points spread out" sounds visual, but it means something practical. If two
# points land on the same shadow spot, the shadow forgot the difference between them. If
# the points stay spread apart, the shadow kept more information about who is who.

# %%
table = pca_hand_table()
table

# %%
print("x spread:", spread(table["x shadow"]))
print("y spread:", spread(table["y shadow"]))

# %% [markdown]
# If we can keep one axis, we keep x. It keeps A and D far apart, while the y shadow
# squishes pairs together. Spread is information because it keeps points different from
# each other.
#
# > 📖 **Grown-ups call this:** **principal component analysis** searches for the shadow
# > where the points stay as spread out as possible.

# %%
workbook.render(21)

# %% [markdown]
# ## 👀 Take a look
#
# The app lets you hunt for the best angle. Here we compare one human-picked shadow with
# PCA's answer.

# %%
X = make_shadow_cloud()
shadow = shadow_projection(X, yaw_deg=0, pitch_deg=0)
pca_shadow, pca_keep = pca_shadow_answer(X)
print("your variance kept:", round(variance_captured(X, shadow), 3))
print("PCA variance kept:", round(pca_keep, 3))

# %%
fig = plot_shadow_2d(shadow, "One possible shadow")
plt.show()
fig = plot_shadow_2d(pca_shadow, "PCA's shadow")
plt.show()

# %% [markdown]
# The best shadow is spread out because spread is information. If every point collapses
# into one spot, the shadow forgot how the points differ. A wider shadow kept more of the
# story. So "variance kept" means "how much of the useful spread survived the squish."

# %% [markdown]
# ## 🎛️ Your turn
#
# Compress one digit to n components and rebuild it. Watch the number turn ghostly, then readable again.

# %%
fig, curve = plot_reconstruction(index=8, n_components=12)
plt.show()
fig = plot_variance_curve(curve, n_components=12)
plt.show()

# %%
fig = plot_eigendigits(8)
plt.show()

# %% [markdown]
# Look at the rebuild first, then the curve. Those ghostly pictures are principal
# components. PCA mixes them to rebuild digits.

# %% [markdown]
# ## 💻 In real code
#
# First, PCA squishes all 8x8 digits to two numbers. The labels only colour the plot
# afterwards; PCA never saw them.

# %%
fig, kept = plot_digits_pca()
plt.show()
print("variance kept:", round(kept, 3))

# %% [markdown]
# Some digits still overlap. That matches the confusion-matrix idea from Chapter 17:
# shared strokes make shared mistakes. Two different tools are agreeing that look-alike
# digits live near each other. Payoff!
#
# t-SNE is a different kind of squishing. It tries to keep neighbours together rather than
# keeping global spread. It bends and stretches the map to make local neighbourhoods
# visible. The gap between two islands, or the size of an island, is not a measured fact.
# People over-read these plots because they look like maps.

# %%
fig = plot_digits_tsne(n=600, seed=0)
plt.show()

# %% [markdown]
# PCA is linear. It can pick a flat shadow, not unwrap every shape. If two curved arms
# cross in every flat shadow, PCA cannot separate them no matter which angle it chooses.

# %%
fig = plot_pca_linear_failure()
plt.show()

# %% [markdown]
# On penguins, the first component mostly acts like a size direction. Look for body-mass
# and flipper-length weights when the table appears.

# %%
loadings, first_kept = penguin_pca_table()
loadings

# %%
print("first-component variance:", round(first_kept, 3))

# %% [markdown]
# ## 🏆 Go further
#
# 1. How many components before you can still read a digit?
# 2. Which two digits stay tangled longest?
# 3. Run PCA on penguins and decide whether the first component is basically size.
# 4. 🧸 **Little Kid Corner side quest:** Make hand shadows. Which angle tells the best story?
#
# ---
# **Next up:** Chapter 22 · *The Bigram Babbler* — where counting letter pairs starts making words.
