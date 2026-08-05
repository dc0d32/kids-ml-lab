"""Chapter 18 · You Are Like Your Neighbors."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kidsml import ui
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

ui.page_setup(18)


@st.cache_data(show_spinner=False)
def cached_timing():
    return knn_timing_table()


@st.cache_data(show_spinner=False)
def cached_penguins(k):
    return penguin_knn_scores(k)


@st.cache_data(show_spinner=False)
def cached_digits(k):
    return digits_knn_score(k)


@st.cache_data(show_spinner=False)
def cached_curve():
    return knn_accuracy_curve()


# ---------------------------------------------------------------------------
ui.beat("hook", "A hinge between two worlds.")

st.markdown(
    """
Every model so far was shown the right answers. Somebody had to label the data.

Part 5 asks a new question: **what can you learn when nobody tells you the answers?**

This chapter is the bridge. kNN still uses labels, so it is not unsupervised. But it does
no training at all. A new point arrives. It looks at nearby old points and copies them.
That is the whole algorithm. It is also embarrassingly good.

This should feel backward. Most models pay a training cost first, then predict quickly.
kNN saves almost all the work until a new point asks for an answer.
"""
)

ui.little_kid_corner(
    "If you move to a new lunch table, you might copy the kids sitting closest to you. "
    "No studying. No notebook. Nearby people vote."
)

ui.mermaid(
    """
graph LR
    A[New point] --> B[Measure distances]
    B --> C[Sort nearest first]
    C --> D[Take k neighbours]
    D --> E[Vote]
    E --> F[Prediction]
""",
    height=240,
)

st.markdown(
    "Follow the arrows slowly. The model is not drawing a formula; it is looking up old cases and asking who is nearby."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Five old points. One new point.")

st.markdown(
    """
The new point is at **(0, 0)**. The old points already have labels.

Distance is the ruler. For point A at (3, 4), the distance is
`√((3 - 0)² + (4 - 0)²) = √(9 + 16) = √25 = 5`. kNN repeats that calculation for every
old point, then sorts the list.
"""
)
st.dataframe(knn_distance_table(), hide_index=True, use_container_width=True)

k_hand = st.select_slider("How many neighbours get to vote?", options=[1, 3, 5], value=3, key="ch18_hand_k")
nearest, votes, winner = knn_vote_table(k_hand)
left, right = st.columns([1, 1])
with left:
    st.markdown("**Nearest voters**")
    st.dataframe(nearest, hide_index=True, use_container_width=True)
with right:
    st.markdown("**Vote tally**")
    st.dataframe(votes, hide_index=True, use_container_width=True)
    st.metric("winner", winner)

ui.aha(
    "Changing **k** can change the answer. With k = 1, one close neighbour has total power. "
    "With k = 5, the wider crowd can overrule it."
)
ui.jargon("k nearest neighbours", "Pick the **k** closest old points, then let them vote.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "The circle reaches the k-th neighbour.")

col_a, col_b = st.columns(2, gap="large")
with col_a:
    fig = plot_knn_hand(k_hand)
    ui.show(fig)
with col_b:
    boundary_k = st.select_slider("Boundary k", options=[1, 3, 5], value=1, key="ch18_boundary_k")
    fig = plot_knn_hand_boundary(boundary_k)
    ui.show(fig)
    st.caption("For k = 1, every region belongs to the nearest point. Grown-ups call this a Voronoi diagram.")
st.markdown(
    "Look at the jagged borders when k is small. That is Chapter 05's deep-tree problem in a new costume: memorising every noisy point can make a wiggly rule."
)

# ---------------------------------------------------------------------------
ui.beat("play", "Place the new point.")

k = st.slider("k", 1, 51, 7, 2, key="ch18_play_k")
qx = st.slider("new point x", -2.5, 2.5, 0.0, 0.1, key="ch18_qx")
qy = st.slider("new point y", -2.0, 2.0, 0.0, 0.1, key="ch18_qy")
fig, votes = plot_knn_play(k=k, qx=qx, qy=qy)
ui.show(fig)
st.dataframe(votes, hide_index=True)

curve = cached_curve()
st.line_chart(curve.set_index("k"), height=260)
st.caption(
    "Look for the middle sweet spot. k = 1 memorises every noisy point. Huge k smooths everything until the whole plane starts giving one answer."
)
ui.careful("An even k can tie in a two-class problem. Odd k does not remove every tie, but it dodges the common one.")

st.markdown("**Catch #1: prediction is the expensive part.** These numbers are measured on this machine.")
st.dataframe(cached_timing().round(2), hide_index=True, use_container_width=True)
st.markdown(
    """
Training is cheap because kNN stores the table and waits. Prediction is costly because one
new point must be compared with many old points. With 100 rows you barely notice. With
10 million rows and many users asking at once, the waiting can bite.
"""
)

st.markdown("**Catch #2: scale matters.** Penguins measured in grams can drown out beaks measured in millimetres.")
st.dataframe(cached_penguins(7).assign(accuracy=lambda d: d["accuracy"].map(lambda x: f"{x:.1%}")), hide_index=True)
st.markdown(
    """
Distance adds feature differences together. A 500-gram body-mass difference can bulldoze a
5-millimetre beak difference because 500 is the bigger number. Change kilograms to
millimetres, or kilograms to grams, and you can change which clue shouts loudest without
changing the animal at all. Scaling puts the columns on fair rulers before neighbours vote.
"""
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "The real sklearn version.")

st.code(
    """
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
model.fit(penguin_measurements, penguin_species)
model.predict(new_penguins)
""",
    language="python",
)

st.metric("8x8 digit accuracy with k = 3", f"{cached_digits(3):.1%}")
st.markdown(
    "That is the fun surprise. The algorithm sounds tiny, but on small images it is strong "
    "because similar-looking digits often sit near each other in pixel-number space."
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find the k that scores best on the moons curve.
2. Find a dataset shape where kNN beats the early straight-line models.
3. Break kNN by multiplying one feature by 1000, then fix it with scaling.
4. 🧸 **Little Kid Corner:** Put toys in two teams. Drop a sock. Let the closest toy vote, then the closest three toys vote. Did the sock switch teams?
"""
)

ui.worksheet_link(18)
