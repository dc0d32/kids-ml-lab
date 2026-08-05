# %% [markdown]
# # Chapter 20 · Squishing Dimensions
#
# ### PCA is picking the best shadow to cast.
#
# *Part 5 · Without answers*
#
# ---
#
# Hold up your hand and cast a shadow on a wall. Turn it. Some shadows tell you it is a
# hand. Other shadows become a blob.
#
# Choosing the angle is the whole of PCA.

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
# ## 🎣 The Hook
#
# Same hand. Same wall. Different angle. The useful shadow keeps the story.
#
# > 🧸 **Little Kid Corner** — Use a flashlight and your hand. Turn your hand until the
# > shadow gives the best clue.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Four points lie nearly sideways. Project onto x, then onto y. Spread is the sum of
# squared distances from the middle.

# %%
table = pca_hand_table()
table

# %%
print("x spread:", spread(table["x shadow"]))
print("y spread:", spread(table["y shadow"]))

# %% [markdown]
# If we can keep one axis, we keep x. It stays more spread out.
#
# > 📖 **Grown-ups call this:** **principal component analysis** searches for the shadow
# > where the points stay as spread out as possible.

# %%
workbook.render(20)

# %% [markdown]
# ## 👀 See It
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
# into one spot, the shadow forgot how the points differ.

# %% [markdown]
# ## 🎛️ Play With It
#
# Compress one digit to n components and rebuild it.

# %%
fig, curve = plot_reconstruction(index=8, n_components=12)
plt.show()
fig = plot_variance_curve(curve, n_components=12)
plt.show()

# %%
fig = plot_eigendigits(8)
plt.show()

# %% [markdown]
# Those ghostly pictures are principal components. PCA mixes them to rebuild digits.

# %% [markdown]
# ## 💻 For Real
#
# First, PCA squishes all 8x8 digits to two numbers. The labels only colour the plot
# afterwards; PCA never saw them.

# %%
fig, kept = plot_digits_pca()
plt.show()
print("variance kept:", round(kept, 3))

# %% [markdown]
# Some digits still overlap. That matches the confusion-matrix idea from Chapter 16: if
# two digits look alike, a small shadow keeps tangling them.
#
# t-SNE is a different kind of squishing. It tries to keep neighbours together rather than
# keeping global spread. It can look cleaner, but distances and cluster sizes in a t-SNE
# plot are not facts.

# %%
fig = plot_digits_tsne(n=600, seed=0)
plt.show()

# %% [markdown]
# PCA is linear. It can pick a flat shadow, not unwrap every shape.

# %%
fig = plot_pca_linear_failure()
plt.show()

# %% [markdown]
# On penguins, the first component mostly acts like a size direction.

# %%
loadings, first_kept = penguin_pca_table()
loadings

# %%
print("first-component variance:", round(first_kept, 3))

# %% [markdown]
# ## 🏆 Challenge
#
# 1. How many components before you can still read a digit?
# 2. Which two digits stay tangled longest?
# 3. Run PCA on penguins and decide whether the first component is basically size.
# 4. 🧸 **Little Kid Corner:** Make hand shadows. Which angle tells the best story?
#
# ---
# **Next up:** Chapter 21 · *The Bigram Babbler* — where counting letter pairs starts making words.
