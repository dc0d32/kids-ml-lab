# %% [markdown]
# # Chapter 08 · You Are Like Your Neighbors
#
# ### The model that does no training at all.
#
# *Part 1 · Classical models*
#
# ---
#
# Every model so far worked the same way. You showed it the answers, it trained, and out
# came a rule: a line, a tree, a road with the widest gap.
#
# This one never trains. It writes down every example it is given, and then it stops.
#
# When a new point turns up, it measures how far that point is from every example it wrote
# down, keeps the closest few, and lets them vote.
#
# The work did not vanish — it moved. Every other model pays its bill up front and then
# predicts fast. This one pays nothing up front and gets the whole bill at prediction time.

# %%
import matplotlib.pyplot as plt

from kidsml import workbook
from kidsml.neighbors import (
    BOUNDARY_KS,
    knn_accuracy_curve,
    plot_knn_accuracy_curve,
    plot_knn_boundary,
)
from kidsml.plots import use_house_style
from kidsml.unsupervised import (
    digits_knn_score,
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
# graph TD
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
# With k = 1, only the single nearest point decides, so one odd point can swing the answer by
# itself. With k = 5, the wider crowd can overrule it. Changing **k** can change the answer.
#
# We can sort by squared distance because square roots keep the same order. If 25 < 36,
# then √25 < √36.
#
# > 📖 **Grown-ups call this:** **k nearest neighbours** means picking the k closest old
# > points and letting them vote.

# %%
workbook.render(8)

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
# Look at the border loop out to fence off single points. Tracing every dot this closely is
# *overfitting* — the same trap the deep tree fell into back in Chapter 5.

# %%
fig = plot_knn_hand_boundary(k=1)
plt.show()

# %% [markdown]
# Five points were easy to picture. Here is the same idea on a real cloud of points, where
# the two clumps overlap in the middle. That overlap is the whole reason **k** matters: right
# along the seam, the number of votes decides the answer.
#
# Watch the boundary morph as k grows: at **k = 1** it loops out to grab single points, around
# **k = 7** it settles into one clean curve, and at **k = 121** it collapses to one flat
# diagonal — the average of everything.

# %%
for k in BOUNDARY_KS:
    fig = plot_knn_boundary(k=k)
    plt.show()

# %% [markdown]
# Now score each k on points the model never trained on. The curve climbs from k = 1, tops
# out in the middle, then falls off a cliff.

# %%
curve = knn_accuracy_curve()
curve

# %%
fig = plot_knn_accuracy_curve(curve)
plt.show()

# %% [markdown]
# Read the hump. **k = 1** memorises the noise, so it does worse on new points than you would
# guess. **Huge k** averages the whole dataset into one sleepy answer and falls off the right
# edge. The best k — marked in green — sits in the middle: big enough to ignore the noise,
# small enough to still see the shape. That middle spot is the whole point of the chapter.
# (An even k can tie a two-team vote, so people usually pick an odd one.)

# %% [markdown]
# ## 🎛️ Your turn
#
# Change the query point and k in the app. Here is one frozen setup to poke.

# %%
fig, votes = plot_knn_play(k=7, qx=0.0, qy=0.0)
plt.show()
votes

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
# Back to the penguins from Chapter 4: 344 real birds measured on three islands. This time
# we want the species, and we hand kNN four measurements: beak length, beak depth, flipper
# length, and body mass.
#
# Penguins give kNN a trap: beaks are measured in millimetres, but body mass is measured
# in grams. Run the raw measurements and the scaled version, then read the gap.

# %%
penguin_knn_scores(k=7)

# %% [markdown]
# Distance adds feature differences together. A 500-gram body-mass difference can
# bulldoze a 5-millimetre beak difference because 500 is the bigger number. **Scaling**
# puts the columns on fair rulers before neighbours vote.

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

# %% [markdown]
# ### Same idea, much longer rows
#
# Every row you have fed this thing so far has been short: two numbers for the toy points,
# four measurements for a penguin. Which raises a fair question — does anything break if a
# row gets *long*?
#
# Here is a row that is not a bird. Scan a handwritten digit into an 8x8 grid and write
# down how bright each little square is. That is 64 numbers in one long row: the same shape
# of thing as the four penguin measurements, just more of them.
#
# Nothing about the algorithm changes. It still measures the distance from the new row to
# every stored row, keeps the closest three, and lets them vote.

# %%
print("handwriting read correctly, k = 3:", round(digits_knn_score(k=3), 3))

# %% [markdown]
# A model that does no training at all reads handwriting this well. Two digits that look
# alike have similar brightness numbers in the same places, so their rows land near each
# other — and near is the only thing this algorithm has ever needed.
#
# Chapter 18 takes pictures seriously. This is just a taste of why it works.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Slide k on the curve above and find the value that scores best, then check it against the green marker.
# 2. Find a dataset shape where kNN beats the straight-line models from Chapters 2 and 7.
# 3. Break kNN by multiplying one feature by 1000, then fix it with scaling.
# 4. 🧸 **Little Kid Corner:** Put toys in two teams. Let one neighbour vote, then three.
#
# ---
# **Next up:** Chapter 09 · *The Model Zoo* — where all these guessers race each other fairly.
