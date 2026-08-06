"""Chapter 08 · You Are Like Your Neighbors."""

from __future__ import annotations

import streamlit as st

from kidsml import lesson
from kidsml.neighbors import (
    BOUNDARY_KS,
    knn_accuracy_curve,
    plot_knn_accuracy_curve,
    plot_knn_boundary,
)
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

lesson.begin(8)


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


@lesson.step("The model that never trains", beat="hook")
def _():
    lesson.say(
        """
Every model so far worked the same way. You showed it the answers, it trained, and out
came a rule: a line, a tree, a road with the widest gap.

This one never trains. It writes down every example it is given, and then it stops. That
is the whole setup step.
"""
    )
    lesson.say(
        """
So what does it do when a new point turns up? It measures how far that point is from
every example it wrote down, keeps the closest few, and lets them vote.

The work did not vanish — it moved. Every other model pays its bill up front and then
predicts fast. This one pays nothing up front and gets the whole bill at prediction time.
"""
    )
    lesson.mermaid(
        """
graph TD
    A[New point] --> B[Measure distances]
    B --> C[Sort nearest first]
    C --> D[Take k neighbours]
    D --> E[Vote]
    E --> F[Prediction]
""",
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
    k_hand = st.select_slider("How many neighbours get to vote?", options=[1, 3, 5], value=3, key="ch08_hand_k")
    nearest, votes, winner = knn_vote_table(k_hand)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Nearest voters**")
        st.dataframe(nearest, hide_index=True, width="stretch")
    with right:
        st.markdown("**Vote tally**")
        st.dataframe(votes, hide_index=True, width="stretch")
        st.metric("winner", winner)
    lesson.look_for("how changing k changes who gets a vote. With k = 1 only the single nearest point decides, so one odd point can swing the whole answer by itself.")
    lesson.jargon("k nearest neighbours", "Pick the **k** closest old points, then let them vote.")


@lesson.step("Predict the k = 1 border", beat="seeit")
def _():
    guess = lesson.predict(
        "With k = 1, what will the boundary look like?",
        ["Smooth and calm", "Jagged, with islands around odd points", "A straight line"],
        correct=1,
        why="Every old point owns the patch of space where it is the nearest neighbour.",
        key="ch08_k1_boundary",
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
    lesson.look_for("the border looping out to fence off single points. Tracing every dot this closely is overfitting — the same trap the deep tree fell into back in Chapter 5.")
    lesson.aha("For k = 1, every region belongs to the nearest point. Grown-ups call this a Voronoi diagram.")


@lesson.step("Morph the boundary", beat="seeit")
def _():
    lesson.say(
        """
Five points were easy to picture. Here is the same idea on a real cloud of points, where
the two clumps overlap in the middle. That overlap is the whole reason **k** matters: right
along the seam, changing how many neighbours vote changes the answer.
"""
    )
    left, right = st.columns(2, gap="large")
    with left:
        boundary_k = st.select_slider(
            "How many neighbours vote?", options=list(BOUNDARY_KS), value=1, key="ch08_boundary_k"
        )
        lesson.show(plot_knn_boundary(boundary_k))
    with right:
        lesson.show(plot_knn_accuracy_curve(cached_curve()))
    lesson.say(
        """
Slide k up and watch the left picture change. At **k = 1** the border loops out to fence off
single points — it has memorised the noise. Around **k = 7** it settles into one clean curve
between the clumps. At **k = 121** it stops seeing the clumps at all and hands back one flat
diagonal: the average of everything.

The right chart scores each of those k values on points the model never trained on. It
climbs from k = 1, tops out in the middle, then falls off a cliff. The best k — marked in
green — is neither the smallest nor the biggest. That middle spot is the whole point of the
chapter.
"""
    )
    lesson.look_for("the green hump on the right. Accuracy is low at k = 1, rises to a peak in the middle, then dives toward the right edge — the exact k values you are sliding through on the left.")
    st.caption("A boundary smoothed all the way to k = 121 has, and I believe I am using this correctly, no aura.")
    lesson.careful("An even k can tie a two-team vote. An odd k does not dodge every tie, but it dodges the most common one.")


@lesson.step("Place the new point", beat="play")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        k = st.slider("k", 1, 51, 7, 2, key="ch08_play_k")
        qx = st.slider("new point x", -2.5, 2.5, 0.0, 0.1, key="ch08_qx")
        qy = st.slider("new point y", -2.0, 2.0, 0.0, 0.1, key="ch08_qy")
    with picture:
        fig, votes = plot_knn_play(k=k, qx=qx, qy=qy)
        lesson.show(fig)
        st.dataframe(votes, hide_index=True)
    lesson.look_for("the star and its lines out to its k nearest points. Drag it across the border and watch the winning colour flip as new neighbours come into range.")


@lesson.step("Predict the timing cost", beat="forreal")
def _():
    guess = lesson.predict(
        "Which part of kNN gets expensive as the remembered table grows?",
        ["Training", "Prediction", "Both stay free"],
        correct=1,
        why="Training stores the table. Prediction has to compare each new point with many old points, ruler after ruler.",
        key="ch08_timing",
    )
    if guess is None:
        return
    st.dataframe(cached_timing().round(2), hide_index=True, width="stretch")
    lesson.look_for("the predict column as rows remembered grows. Saving work during training slides the bill to prediction time.")


@lesson.step("Predict the scale disaster", beat="forreal")
def _():
    lesson.say(
        """
Back to the penguins from Chapter 4 — the 344 real birds measured on three islands. This
time we want the species, and we hand kNN four measurements instead of two: beak length,
beak depth, flipper length and body mass.

Here is the trap. kNN decides everything by **distance**, and those four numbers are not
in the same units.
"""
    )

    guess = lesson.predict(
        "Penguins have beaks in millimetres and body mass in grams. What can go wrong?",
        ["Grams can drown out beak lengths", "All columns get equal voice", "Distances stop mattering"],
        correct=0,
        why="A 500-gram body-mass difference can bulldoze a 5-millimetre beak difference because 500 is the bigger number.",
        key="ch08_scale",
    )
    if guess is None:
        return
    st.dataframe(cached_penguins(7).assign(accuracy=lambda d: d["accuracy"].map(lambda x: f"{x:.1%}")), hide_index=True)
    lesson.look_for("raw measurements versus scaled first. Scaling puts the columns on fair rulers before neighbours vote.")
    lesson.jargon("scaling", "Putting columns onto comparable rulers before distance gets measured.")


@lesson.step("The sklearn version", beat="forreal")
def _():
    lesson.say(
        "In real code the fix is one extra step in front of the model. `StandardScaler` "
        "puts every column on a fair ruler first, and the pipeline makes sure that happens "
        "every time — including on penguins the model has never seen."
    )
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
    lesson.look_for("the scaler sitting in front of the model. The ruler-fixing happens first, every single time, including on birds the model has never seen.")


@lesson.step("Same idea, much longer rows", beat="forreal")
def _():
    lesson.say(
        """
Every row you have fed this thing so far has been short. Two numbers for the toy points,
four measurements for a penguin. Which raises a fair question: does anything break if a
row gets *long*?
"""
    )
    lesson.say(
        """
Here is a row that is not a bird. Scan a handwritten digit into an 8-by-8 grid and write
down how bright each little square is. That is 64 numbers, laid out in one long row —
exactly the same shape of thing as the four penguin measurements, just more of them.

Nothing about the algorithm changes. It still measures the distance from your new row to
every stored row, keeps the closest three, and lets them vote.
"""
    )
    st.metric("Handwriting read correctly, k = 3", f"{cached_digits(3):.1%}")
    lesson.aha(
        "A model that does no training at all reads handwriting this well. Two digits that "
        "look alike have similar brightness numbers in the same places, so their rows land "
        "near each other — and near is the only thing this algorithm has ever needed."
    )
    lesson.say("Chapter 18 takes pictures seriously. This is just a taste of why it works.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. Slide the boundary k until you find the value that scores best on the curve, then check it against the green marker.
2. Find a dataset shape where kNN beats the straight-line models from Chapters 2 and 7.
3. Break kNN by multiplying one feature by 1000, then fix it with scaling.
"""
    )
    lesson.kid_corner("Put toys in two teams. Drop a sock. Let the closest toy vote, then the closest three toys vote. Did the sock switch teams?")


lesson.finish()
