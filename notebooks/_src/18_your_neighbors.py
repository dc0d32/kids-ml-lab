# %% [markdown]
# # Chapter 18 · You Are Like Your Neighbors
#
# ### The model that does no training at all.
#
# *Part 5 · Without answers*
#
# ---
#
# Every model so far was shown the right answers. Somebody labelled the data.
#
# Part 5 asks what you can learn when nobody tells you the answers. This chapter is the
# bridge: kNN still uses labels, so it is not unsupervised. But it skips training.

# %%
import matplotlib.pyplot as plt

from kidsml import workbook
from kidsml.plots import use_house_style
from kidsml.unsupervised import (
    digits_knn_score,
    knn_accuracy_curve,
    knn_distance_table,
    knn_timing_table,
    knn_vote_table,
    penguin_knn_scores,
    plot_knn_hand,
    plot_knn_hand_boundary,
    plot_knn_play,
)

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# A new point arrives. Look at the closest old points. Copy their answer.
#
# That is kNN. It sounds tiny. It is embarrassingly good.
#
# > 🧸 **Little Kid Corner** — Drop a sock near toys from two teams. The closest toys vote
# > on which team the sock joins.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# The new point is at **(0, 0)**. The old points are labelled.

# %%
distances = knn_distance_table()
distances

# %%
for k in [1, 3, 5]:
    nearest, votes, winner = knn_vote_table(k)
    print(f"k = {k}: {winner} wins")
    display(nearest[["point", "label", "distance"]])

# %% [markdown]
# The answer changes: k = 1 says red, k = 3 says blue, k = 5 says red again.
#
# We can sort by squared distance because square roots keep the same order. If 25 < 36,
# then √25 < √36.
#
# > 📖 **Grown-ups call this:** **k nearest neighbours** means picking the k closest old
# > points and letting them vote.

# %%
workbook.render(18)

# %% [markdown]
# ## 👀 See It
#
# The dashed circle grows until it touches the k-th neighbour.

# %%
fig = plot_knn_hand(k=3)
plt.show()

# %% [markdown]
# For k = 1, every patch of the plane belongs to the nearest point. That picture has a
# grown-up name: a Voronoi diagram.

# %%
fig = plot_knn_hand_boundary(k=1)
plt.show()

# %% [markdown]
# ## 🎛️ Play With It
#
# Change the query point and k in the app. Here is one frozen setup.

# %%
fig, votes = plot_knn_play(k=7, qx=0.0, qy=0.0)
plt.show()
votes

# %%
curve = knn_accuracy_curve()
curve.head()

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(curve["k"], curve["test accuracy"], marker="o")
ax.set_xlabel("k")
ax.set_ylabel("test accuracy")
ax.set_title("The sweet spot is usually in the middle")
plt.show()

# %% [markdown]
# k = 1 memorises noisy points. Huge k smooths so much that the whole plane starts giving
# one answer. Even k can tie in a two-class vote.

# %% [markdown]
# ### Catch #1: prediction is the expensive part
#
# kNN fit is tiny because it stores the data. Prediction has to compare against the stored
# points. These are real measurements from this run.

# %%
timing = knn_timing_table()
timing.round(2)

# %% [markdown]
# ### Catch #2: scale matters
#
# If one column is in grams and another in millimetres, the big-number column can dominate
# distance.

# %%
penguin_knn_scores(k=7)

# %% [markdown]
# ## 💻 For Real
#
# ```python
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
#
# model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
# model.fit(penguin_measurements, penguin_species)
# model.predict(new_penguins)
# ```

# %%
print("8x8 digit accuracy with k = 3:", round(digits_knn_score(k=3), 3))

# %% [markdown]
# That is the surprise. kNN sounds tiny, but on the small digit images it is strong.

# %% [markdown]
# ## 🏆 Challenge
#
# 1. Find the k that scores best on moons.
# 2. Find a dataset shape where kNN beats the early straight-line models.
# 3. Break kNN by multiplying one feature by 1000, then fix it with scaling.
# 4. 🧸 **Little Kid Corner:** Put toys in two teams. Let one neighbour vote, then three.
#
# ---
# **Next up:** Chapter 19 · *Sorting Without Labels* — where the labels disappear.
