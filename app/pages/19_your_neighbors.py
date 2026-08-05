"""Chapter 19 · You Are Like Your Neighbors."""

from __future__ import annotations

import streamlit as st

from kidsml import lesson
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

lesson.begin(19)


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


@lesson.step("A hinge between two worlds", beat="hook")
def _():
    lesson.say(
        """
Every model so far was shown the right answers. Somebody had to label the data.

Part 5 opens a new door: **what can you learn when nobody tells you the answers?**
"""
    )
    lesson.mermaid(
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
    lesson.look_for("the delayed work. kNN stores old cases, then pulls out the ruler when a new point asks for an answer.")
    lesson.kid_corner("If you move to a new lunch table, you might copy the kids sitting closest to you. No studying. Nearby people vote.")


@lesson.step("Five old points, one new point", beat="byhand")
def _():
    lesson.say(
        """
The new point is at **(0, 0)**. The old points already have labels.

Distance is the ruler. For point A at (3, 4), the distance is
`√((3 - 0)² + (4 - 0)²) = √(9 + 16) = √25 = 5`. Nice 3-4-5 triangle, ready to vote.
"""
    )
    st.dataframe(knn_distance_table(), hide_index=True, width="stretch")
    lesson.look_for("the distance column. kNN repeats the ruler move for every old point, then sorts the list.")


@lesson.step("Let the neighbours vote", beat="byhand")
def _():
    k_hand = st.select_slider("How many neighbours get to vote?", options=[1, 3, 5], value=3, key="ch19_hand_k")
    nearest, votes, winner = knn_vote_table(k_hand)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Nearest voters**")
        st.dataframe(nearest, hide_index=True, width="stretch")
    with right:
        st.markdown("**Vote tally**")
        st.dataframe(votes, hide_index=True, width="stretch")
        st.metric("winner", winner)
    lesson.look_for("how changing k changes who gets a vote. One close neighbour can grab the whole microphone.")
    lesson.jargon("k nearest neighbours", "Pick the **k** closest old points, then let them vote.")


@lesson.step("Predict the k = 1 border", beat="seeit")
def _():
    guess = lesson.predict(
        "With k = 1, what will the boundary look like?",
        ["Smooth and calm", "Jagged, with islands around odd points", "A straight line"],
        correct=1,
        why="Every old point owns the patch of space where it is the nearest neighbour.",
        key="ch19_k1_boundary",
    )
    if guess is None:
        return

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        fig = plot_knn_hand(1)
        lesson.show(fig)
    with col_b:
        fig = plot_knn_hand_boundary(1)
        lesson.show(fig)
    lesson.look_for("the jagged borders. This is Chapter 05's overfitting wearing a different hat.")
    lesson.aha("For k = 1, every region belongs to the nearest point. Grown-ups call this a Voronoi diagram.")


@lesson.step("Morph the boundary", beat="seeit")
def _():
    boundary_k = st.select_slider("Boundary k", options=[1, 3, 5], value=1, key="ch19_boundary_k")
    fig = plot_knn_hand_boundary(boundary_k)
    lesson.show(fig)
    lesson.look_for("what happens as k grows. The wider crowd smooths the tiny islands.")
    lesson.careful("An even k can tie in a two-class problem. Odd k does not remove every tie, but it dodges the common one.")


@lesson.step("Place the new point", beat="play")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        k = st.slider("k", 1, 51, 7, 2, key="ch19_play_k")
        qx = st.slider("new point x", -2.5, 2.5, 0.0, 0.1, key="ch19_qx")
        qy = st.slider("new point y", -2.0, 2.0, 0.0, 0.1, key="ch19_qy")
    with picture:
        fig, votes = plot_knn_play(k=k, qx=qx, qy=qy)
        lesson.show(fig)
        st.dataframe(votes, hide_index=True)
    lesson.look_for("the star and its vote lines. Drag the star across the border and watch whose votes storm in.")


@lesson.step("There is a sweet spot", beat="play")
def _():
    curve = cached_curve()
    st.line_chart(curve.set_index("k"), height=260)
    lesson.look_for("the middle sweet spot. k = 1 memorises every noisy point; huge k smooths until the plane starts giving one sleepy answer.")


@lesson.step("Predict the timing cost", beat="forreal")
def _():
    guess = lesson.predict(
        "Which part of kNN gets expensive as the remembered table grows?",
        ["Training", "Prediction", "Both stay free"],
        correct=1,
        why="Training stores the table. Prediction has to compare each new point with many old points, ruler after ruler.",
        key="ch19_timing",
    )
    if guess is None:
        return
    st.dataframe(cached_timing().round(2), hide_index=True, width="stretch")
    lesson.look_for("the predict column as rows remembered grows. Saving work during training slides the bill to prediction time.")


@lesson.step("Predict the scale disaster", beat="forreal")
def _():
    guess = lesson.predict(
        "Penguins have beaks in millimetres and body mass in grams. What can go wrong?",
        ["Grams can drown out beak lengths", "All columns get equal voice", "Distances stop mattering"],
        correct=0,
        why="A 500-gram body-mass difference can bulldoze a 5-millimetre beak difference because 500 is the bigger number.",
        key="ch19_scale",
    )
    if guess is None:
        return
    st.dataframe(cached_penguins(7).assign(accuracy=lambda d: d["accuracy"].map(lambda x: f"{x:.1%}")), hide_index=True)
    lesson.look_for("raw measurements versus scaled first. Scaling puts the columns on fair rulers before neighbours vote.")


@lesson.step("The sklearn version", beat="forreal")
def _():
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
    lesson.look_for("the scaler in the pipeline. The ruler-fixing step happens before the neighbour vote.")
    lesson.aha("The algorithm sounds tiny, but on small images it hits hard because similar-looking digits often sit near each other in pixel-number space!")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. Find the k that scores best on the moons curve.
2. **No cap neighbour test.** Find a dataset shape where kNN beats the early straight-line models.
3. Break kNN by multiplying one feature by 1000, then fix it with scaling.
"""
    )
    lesson.kid_corner("Put toys in two teams. Drop a sock. Let the closest toy vote, then the closest three toys vote. Did the sock switch teams?")


lesson.finish()
