# %% [markdown]
# # Chapter 20 · Sorting Without Labels
#
# ### k-means — and your photo squeezed to 5 colours.
#
# *Part 5 · Without answers*
#
# ---
#
# Every model so far was shown the right answers. Somebody had to label the data.
#
# Part 5 asks a different question: what can you learn when nobody tells you the answers?
#
# Nobody labels anything this time. Here is a pile of dots. Find the clumps.
#
# The danger is that "clump" sounds like a thing everyone will agree on. Sometimes they
# will. Sometimes one person's two piles are another person's three piles. The dots do not
# wear name tags. Clustering is useful, but it does not remove judgement.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image

from kidsml import kmeansanim, workbook
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
# ## 🎣 Start here
#
# You already sort without labels. Laundry becomes piles before anyone gives the piles
# fancy names.
#
# > 🧸 **Little Kid Corner** — Put blocks on the floor. Move nearby blocks into piles.
# > Name the piles after you see them.
#
# > 📖 **Grown-ups call this:** **clustering** means sorting data into groups when answer
# > labels are missing.
# > A **cluster** is one of those groups: points that ended up in the same pile.

# %% [markdown]
# ## ✏️ Work it out
#
# k-means has two steps:
#
# 1. Every point joins the nearest centre.
# 2. Every centre scoots to the middle of its members.
#
# Repeat until nothing changes. That stopping point is guaranteed to arrive.
#
# Why must it stop? Each round either makes the total squared point-to-centre distance smaller, or
# nothing changes. There are only so many possible assignments of points to centres. The
# loop cannot keep finding a new smaller setup forever.
#
# ```mermaid
# graph TD
#     A[Choose centres] --> B[Assign points]
#     B --> C[Move centres]
#     C --> D{Anything changed?}
#     D -->|yes| B
#     D -->|no| E[Stop]
# ```
#
# Watch the loop: assign, move, check. The same two-step dance repeats until the centres
# stop moving.

# %%
X, centres = kmeans_hand_points()
points = pd.DataFrame(X, columns=["x", "y"])
points.insert(0, "point", [f"P{i}" for i in range(1, 7)])
points

# %%
pd.DataFrame(centres, columns=["x", "y"], index=["left", "right"])

# %%
assignments, new_centres = kmeans_hand_round()
assignments

# %%
new_centres

# %% [markdown]
# The left centre moves to `((1+1+2)/3, (1+2+1)/3) = (1.33, 1.33)`. The right centre
# moves to `((7+8+7)/3, (7+7+8)/3) = (7.33, 7.33)`. Those numbers match the table.
# Round two changes nothing. The centres have converged!
#
# > 📖 **Grown-ups call this:** **k-means** means using k moving centres to make k piles.
# > A **centroid** is the middle of a cluster. k-means centres move to the centroid of
# > their points.

# %%
workbook.render(20)

# %% [markdown]
# ## 👀 Take a look
#
# The algorithm is *named* after the centres moving to the mean, so watch them do it. In this
# clip every dot recolours to its nearest centre, then the X markers glide to the middle of
# their group, then it repeats — until the title says the centres have settled and nothing
# moves. The two half-steps have their own captions so you can tell them apart.

# %%
Image(data=kmeansanim.kmeans_run_gif_bytes(k=3, seed=0))

# %% [markdown]
# Prefer to freeze it? The app has a step button. Here are four steps from the same run.

# %%
history = kmeans_history(k=3, seed=0)
for stage in history[:4]:
    fig = plot_kmeans_stage(stage)
    plt.show()

# %% [markdown]
# Bad starts can trap k-means. Two centres can begin in the same clump and waste a centre,
# so one real clump may never get its own centre. A bad start can settle into that mistake
# and never fix itself.
#
# That explains sklearn's defaults. k-means++ spreads the starting centres out on purpose,
# then sklearn tries several starts and keeps the best result. It is not magic; it is a
# defence against unlucky first guesses.

# %%
bad_history = kmeans_history(k=3, seed=0, bad_start=True)
fig = plot_kmeans_stage(bad_history[-1])
plt.show()

# %% [markdown]
# ## 🎛️ Your turn
#
# How many clumps should there be? The honest answer is: you judge it.
#
# Inertia means total squared distance from every point to its own centre. It always falls
# as k rises because adding a centre gives the algorithm another bucket for points. It can
# keep the old setup or improve it.
#
# That makes "minimise inertia" useless advice by itself. With one centre per point,
# inertia hits zero and the clusters teach you nothing. The elbow asks where the
# improvement stops earning its keep.

# %%
fig = plot_elbow("obvious")
plt.show()
fig = plot_elbow("ambiguous")
plt.show()

# %% [markdown]
# ### Squeeze a photo to five colours
#
# Treat every pixel as a point in 3D colour space: `(red amount, green amount, blue
# amount)`. The pixel at row 10, column 20 becomes one dot in colour-world. Fit on a sample
# of pixels, then repaint every pixel with its nearest centre colour. Look at the palette
# afterward: those are the colours that survived the squeeze.

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
# ## 💻 In real code
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
# Now peel off the penguin species labels. k-means largely rediscovers them, but cluster
# numbers are arbitrary. Cluster 0 is not a species name.

# %%
table, score = penguin_kmeans_table()
table

# %%
print("cluster/species agreement:", round(score, 3))

# %% [markdown]
# ## 🏆 Go further
#
# 1. Find a seed that makes k-means fail on easy blobs.
# 2. **Elbow hunt.** Find the k where the elbow is clearest.
# 3. Quantize a photo down to 2 colours. Is it still recognisable?
# 4. 🧸 **Little Kid Corner:** Sort laundry into two piles, then three.
#
# ---
# **Next up:** Chapter 21 · *Squishing Dimensions* — where data becomes a shadow.
