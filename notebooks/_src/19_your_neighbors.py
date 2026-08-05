# %% [markdown]
# # Chapter 19 · You Are Like Your Neighbors
#
# ### The model that does no training at all.
#
# *Part 5 · Without answers*
#
# ---
#
# Every model so far was shown the right answers. Somebody labelled the data.
#
# Part 5 opens a new door: what can you learn when nobody tells you the answers? This
# chapter is the bridge: kNN still uses labels, so it is not unsupervised. But it skips
# training.
#
# This should feel backward. Most models pay a training cost first, then predict quickly.
# kNN saves almost all the work until a new point knocks and asks for an answer.

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
# ## 🎣 Start here
#
# A new point arrives. Look at the closest old points. Copy their answer.
#
# That is kNN. It sounds tiny. It is embarrassingly good.
#
# ```mermaid
# graph LR
#     A[New point] --> B[Measure distances]
#     B --> C[Sort nearest first]
#     C --> D[Take k neighbours]
#     D --> E[Vote]
#     E --> F[Prediction]
# ```
#
# Follow the arrows slowly. The model is not drawing a formula; it is pulling out a ruler,
# looking up old cases, and asking who is nearby.
#
# > 🧸 **Little Kid Corner** — Drop a sock near toys from two teams. The closest toys vote
# > on which team the sock joins.

# %% [markdown]
# ## ✏️ Work it out
#
# The new point is at **(0, 0)**. The old points are labelled.
#
# Distance is the ruler. For point A at (3, 4), the distance is
# `√((3 - 0)² + (4 - 0)²) = √(9 + 16) = √25 = 5`. Nice 3-4-5 triangle. kNN repeats that
# ruler move for every old point, then sorts the list.

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
# With k = 1, one close neighbour grabs the whole microphone. With k = 5, the wider crowd
# can overrule it. Changing **k** can change the answer.
#
# We can sort by squared distance because square roots keep the same order. If 25 < 36,
# then √25 < √36.
#
# > 📖 **Grown-ups call this:** **k nearest neighbours** means picking the k closest old
# > points and letting them vote.

# %%
workbook.render(19)

# %% [markdown]
# ## 👀 Take a look
#
# The dashed circle grows until it touches the k-th neighbour. That circle is the voting room.

# %%
fig = plot_knn_hand(k=3)
plt.show()

# %% [markdown]
# For k = 1, every patch of the plane belongs to the nearest point. That picture has a
# grown-up name: a Voronoi diagram.
#
# Look at the jagged borders when k is small. That is Chapter 05's deep-tree problem in a
# new costume: memorising every noisy point can make a wiggly rule.

# %%
fig = plot_knn_hand_boundary(k=1)
plt.show()

# %% [markdown]
# ## 🎛️ Your turn
#
# Change the query point and k in the app. Here is one frozen setup to poke.

# %%
fig, votes = plot_knn_play(k=7, qx=0.0, qy=0.0)
plt.show()
votes

# %%
curve = knn_accuracy_curve()
curve

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(curve["k"], curve["test accuracy"], marker="o")
ax.set_xlabel("k")
ax.set_ylabel("test accuracy")
ax.set_title("The sweet spot is usually in the middle")
plt.show()

# %% [markdown]
# Look for the middle sweet spot. k = 1 memorises noisy points. Huge k smooths so much
# that the whole plane starts giving one sleepy answer. Even k can tie in a two-class vote.

# %% [markdown]
# ### Catch #1: prediction is the expensive part
#
# kNN fit is tiny because it stores the data. Prediction has to compare against the stored
# points. With 100 rows you barely notice. With 10 million rows and many users asking at
# once, the waiting can bite. These are real measurements from this run.

# %%
timing = knn_timing_table()
timing.round(2)

# %% [markdown]
# ### Catch #2: scale matters
#
# Penguins give kNN a trap: beaks are measured in millimetres, but body mass is measured
# in grams. Run the raw measurements and the scaled version, then read the gap.

# %%
penguin_knn_scores(k=7)

# %% [markdown]
# Distance adds feature differences together. A 500-gram body-mass difference can
# bulldoze a 5-millimetre beak difference because 500 is the bigger number. Scaling puts
# the columns on fair rulers before neighbours vote.

# %% [markdown]
# ## 💻 In real code
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
# That is the surprise. kNN sounds tiny, but on small digit images it hits hard because
# similar-looking digits often sit near each other in pixel-number space!

# %% [markdown]
# ## 🏆 Go further
#
# 1. Find the k that scores best on moons.
# 2. **No cap neighbour test.** Find a dataset shape where kNN beats the early straight-line models.
# 3. Break kNN by multiplying one feature by 1000, then fix it with scaling.
# 4. 🧸 **Little Kid Corner:** Put toys in two teams. Let one neighbour vote, then three.
#
# ---
# **Next up:** Chapter 20 · *Sorting Without Labels* — where the labels disappear.
