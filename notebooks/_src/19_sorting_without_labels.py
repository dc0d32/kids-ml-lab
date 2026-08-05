# %% [markdown]
# # Chapter 19 · Sorting Without Labels
#
# ### k-means — and your photo squeezed to 5 colours.
#
# *Part 5 · Without answers*
#
# ---
#
# Nobody labels anything this time. Here is a pile of dots. Find the clumps.
#
# The danger is that "clump" sounds like a thing everyone will agree on. Sometimes they
# will. Sometimes one person's two piles are another person's three piles. Clustering is
# useful, but it does not remove judgement.

# %%
import matplotlib.pyplot as plt

from kidsml import workbook
from kidsml.plots import use_house_style
from kidsml.unsupervised import (
    default_flower_image,
    kmeans_elbow_data,
    kmeans_hand_points,
    kmeans_hand_round,
    kmeans_history,
    penguin_kmeans_table,
    plot_dbscan_moons,
    plot_elbow,
    plot_kmeans_failure,
    plot_kmeans_stage,
    plot_palette,
    quantize_image,
)

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# You already sort without labels. Laundry becomes piles before anyone gives the piles
# fancy names.
#
# > 🧸 **Little Kid Corner** — Put blocks on the floor. Move nearby blocks into piles.
# > Name the piles after you see them.
#
# > 📖 **Grown-ups call this:** **clustering** means sorting data into groups when answer
# > labels are missing.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# k-means has two steps:
#
# 1. Every point joins the nearest centre.
# 2. Every centre moves to the middle of its members.
#
# Repeat until nothing changes. That stopping point is guaranteed to arrive.
#
# Why must it stop? Each round either makes the total point-to-centre distance smaller, or
# nothing changes. There are only so many possible assignments of points to centres. You
# cannot keep finding a new smaller setup forever.
#
# ```mermaid
# graph LR
#     A[Choose centres] --> B[Assign points]
#     B --> C[Move centres]
#     C --> D{Anything changed?}
#     D -->|yes| B
#     D -->|no| E[Stop]
# ```
#
# Watch the loop: assign, move, check. The same two steps repeat until the centres stop
# moving.

# %%
X, centres = kmeans_hand_points()
print("points")
print(X)
print("starting centres")
print(centres)

# %%
assignments, new_centres = kmeans_hand_round()
assignments

# %%
new_centres

# %% [markdown]
# Round two changes nothing. The centres have converged.
#
# > 📖 **Grown-ups call this:** **k-means** means using k moving centres to make k piles.

# %%
workbook.render(19)

# %% [markdown]
# ## 👀 See It
#
# The app has a step button. Here are four steps from one run.

# %%
history = kmeans_history(k=3, seed=0)
for stage in history[:4]:
    fig = plot_kmeans_stage(stage)
    plt.show()

# %% [markdown]
# Bad starts can trap k-means. Two centres can begin in the same clump and waste a centre,
# so one real clump may never get its own centre.
#
# That explains sklearn's defaults. k-means++ spreads the starting centres out on purpose,
# then sklearn tries several starts and keeps the best result. It is not magic; it is a
# defence against unlucky first guesses.

# %%
bad_history = kmeans_history(k=3, seed=0, bad_start=True)
fig = plot_kmeans_stage(bad_history[-1])
plt.show()

# %% [markdown]
# ## 🎛️ Play With It
#
# How many clumps should there be? The honest answer is: you judge it.
#
# Inertia means total distance from every point to its own centre. It always falls as k
# rises because adding a centre gives the algorithm another place to put points. It can
# keep the old setup or improve it.
#
# That makes "minimise inertia" useless advice by itself. With one centre per point,
# inertia hits zero and the clusters teach you nothing. The elbow asks where the
# improvement stops being worth the extra pile.

# %%
fig = plot_elbow("obvious")
plt.show()
fig = plot_elbow("ambiguous")
plt.show()

# %% [markdown]
# ### Squeeze a photo to five colours
#
# Treat every pixel as a point in 3D colour space: red, green, blue. Fit on a sample of
# pixels, then repaint every pixel with its centre colour. Look at the palette afterward:
# those are the colours that survived the squeeze.

# %%
image = default_flower_image()
rebuilt, palette = quantize_image(image, k=5)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].imshow(image)
axes[0].set_title("original")
axes[1].imshow(rebuilt)
axes[1].set_title("5 colours")
for ax in axes:
    ax.set_axis_off()
plt.show()

# %%
fig = plot_palette(palette)
plt.show()

# %% [markdown]
# ## 💻 For Real
#
# k-means likes round-ish, similar-sized blobs because each centre owns the points closest
# to it, making straight-ish borders. Crescents need a curved border, so k-means slices
# through them instead of following the moon shape. DBSCAN handles the crescent shape
# better, and we leave it there.

# %%
fig = plot_kmeans_failure("moons")
plt.show()
fig = plot_kmeans_failure("circles")
plt.show()
fig = plot_dbscan_moons()
plt.show()

# %% [markdown]
# Now hide the penguin species labels. k-means largely rediscovers them, but cluster
# numbers are arbitrary. Cluster 0 is not a species name.

# %%
table, score = penguin_kmeans_table()
table

# %%
print("cluster/species agreement:", round(score, 3))

# %% [markdown]
# ## 🏆 Challenge
#
# 1. Find a seed that makes k-means fail on easy blobs.
# 2. Find the k where the elbow is clearest.
# 3. Quantize a photo down to 2 colours. Is it still recognisable?
# 4. 🧸 **Little Kid Corner:** Sort laundry into two piles, then three.
#
# ---
# **Next up:** Chapter 20 · *Squishing Dimensions* — where data becomes a shadow.
