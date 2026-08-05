"""Chapter 19 · Sorting Without Labels."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml.unsupervised import (
    default_flower_image,
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
    uploaded_image_to_array,
)

lesson.begin(19)


@st.cache_data(show_spinner=False)
def cached_default_image():
    return default_flower_image()


@st.cache_data(show_spinner=False)
def cached_quantized(image, k):
    return quantize_image(image, k=k)


@st.cache_data(show_spinner=False)
def cached_penguin_clusters():
    return penguin_kmeans_table()


@lesson.step("Find piles without labels", beat="hook")
def _():
    lesson.say(
        """
Nobody labels anything this time.

Here is a pile of dots. Find the clumps. You already do this when you sort laundry into
piles without anyone telling you the pile names.
"""
    )
    lesson.careful(
        "The danger is that 'clump' sounds like a thing everyone will agree on. Sometimes one person's two piles are another person's three piles."
    )
    lesson.kid_corner("Put blocks on the floor. Move them into piles by what feels near. You do not need names for the piles before you start.")
    lesson.jargon("clustering", "Sorting data into groups when the answer labels are missing.")


@lesson.step("Two steps you can recite", beat="byhand")
def _():
    lesson.say("**Step 1:** every point joins the nearest centre. **Step 2:** every centre moves to the middle of its members. Repeat until nothing changes.")
    lesson.mermaid(
        """
graph LR
    A[Choose centres] --> B[Assign points]
    B --> C[Move centres]
    C --> D{Anything changed?}
    D -->|yes| B
    D -->|no| E[Stop]
""",
        height=260,
    )
    lesson.look_for("the loop: assign, move, check. The same two steps repeat until the centres stop moving.")
    lesson.say("Why must it stop? Each round either makes total point-to-centre distance smaller, or nothing changes. There are only so many possible assignments.")


@lesson.step("One round by hand", beat="byhand")
def _():
    X, centres = kmeans_hand_points()
    assignments, new_centres = kmeans_hand_round()
    left, right = st.columns(2)
    with left:
        st.markdown("**Six points**")
        st.dataframe(pd.DataFrame(X, columns=["x", "y"]), hide_index=True)
        st.markdown("**Starting centres**")
        st.dataframe(pd.DataFrame(centres, columns=["x", "y"]), hide_index=True)
    with right:
        st.markdown("**Assign points**")
        st.dataframe(assignments, hide_index=True, use_container_width=True)
        st.markdown("**Move centres**")
        st.dataframe(new_centres, hide_index=True, use_container_width=True)
    lesson.look_for("which centre each point joins, then where the new centre lands.")
    lesson.aha("Round two changes nothing. That means the algorithm has converged: it stopped moving.")
    lesson.jargon("k-means", "Pick k centres, assign points to nearest centres, move centres to group middles, and repeat.")


@lesson.step("Press the step button", beat="seeit")
def _():
    k = st.slider("How many centres?", 2, 5, 3, key="ch19_k")
    seed = st.slider("Starting-position seed", 0, 12, 0, key="ch19_seed")
    bad = st.checkbox("Start two centres inside the same clump", value=False, key="ch19_bad")
    settings = (k, seed, bad)
    if st.session_state.get("ch19_settings") != settings:
        st.session_state["ch19_settings"] = settings
        st.session_state["ch19_step"] = 0
    history = kmeans_history(k=k, seed=seed, bad_start=bad)
    if st.button("One k-means step ▶", key="ch19_step_button"):
        st.session_state["ch19_step"] = (st.session_state.get("ch19_step", 0) + 1) % len(history)
    step = st.session_state.get("ch19_step", 0)
    fig = plot_kmeans_stage(history[step])
    lesson.show(fig)
    lesson.look_for("the black X markers. Assign steps recolour dots; move steps shift the centres.")
    st.caption(history[step]["caption"])


@lesson.step("The bad-start trap", beat="seeit")
def _():
    guess = lesson.predict(
        "If two centres begin inside the same real clump, what can happen?",
        ["One real clump may never get a centre", "The algorithm fixes it every time", "All centres vanish"],
        correct=0,
        why="k-means only improves from its starting guess. A bad start can settle into a bad but stable setup.",
        key="ch19_bad_start",
    )
    if guess is None:
        return
    history = kmeans_history(k=3, seed=0, bad_start=True)
    step = st.slider("Bad-start stage", 0, len(history) - 1, len(history) - 1, key="ch19_bad_stage")
    fig = plot_kmeans_stage(history[step])
    lesson.show(fig)
    lesson.look_for("the clump that shares or misses a centre. Bad starts can get stuck.")
    lesson.say("That explains sklearn's defaults. k-means++ spreads the starting centres out on purpose, then sklearn tries several starts and keeps the best result.")


@lesson.step("How many clumps?", beat="play")
def _():
    left, right = st.columns(2, gap="large")
    with left:
        lesson.show(plot_elbow("obvious"))
    with right:
        lesson.show(plot_elbow("ambiguous"))
    lesson.look_for("where the line stops dropping fast. The sharp plot argues for k = 3; the mushy one asks for judgement.")
    lesson.say("Inertia always falls as k rises because adding a centre gives the algorithm another place to put points.")
    lesson.careful("With one centre per point, inertia hits zero and the clusters teach you nothing. The elbow asks where the extra pile stops being worth it.")


@lesson.step("Predict five colours", beat="play")
def _():
    guess = lesson.predict(
        "If a photo is repainted with only five colours, what do you expect?",
        ["It stays recognisable but poster-like", "It becomes random noise", "It becomes a perfect copy"],
        correct=0,
        why="k-means chooses a small palette, then repaints every pixel with its nearest palette colour.",
        key="ch19_five_colours",
    )
    if guess is None:
        return
    uploaded = st.file_uploader("Upload a photo if you want. If not, we use sklearn's flower.", type=["png", "jpg", "jpeg"], key="ch19_upload")
    image = uploaded_image_to_array(uploaded) if uploaded is not None else cached_default_image()
    colour_k = st.slider("palette size", 2, 16, 5, key="ch19_palette")
    rebuilt, palette = cached_quantized(image, colour_k)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.image(image, caption="original", use_container_width=True)
    with col_b:
        st.image(rebuilt, caption=f"{colour_k}-colour version", use_container_width=True)
    lesson.look_for("which colours survived in the rebuilt photo. The image keeps shape while losing many tiny colour differences.")
    lesson.show(plot_palette(palette))
    lesson.look_for("the palette swatches. k-means found these colours from sampled pixels, then repainted every pixel.")


@lesson.step("Predict the moon failure", beat="forreal")
def _():
    guess = lesson.predict(
        "Will k-means separate two crescent moons cleanly?",
        ["Yes, because there are two groups", "No, because centres make round-ish chunks", "Only if k is 10"],
        correct=1,
        why="Each centre owns nearby points, making straight-ish borders. Crescents need a curved border.",
        key="ch19_moons_fail",
    )
    if guess is None:
        return
    left, right, third = st.columns(3, gap="large")
    with left:
        lesson.show(plot_kmeans_failure("moons"))
    with right:
        lesson.show(plot_kmeans_failure("circles"))
    with third:
        lesson.show(plot_dbscan_moons())
    lesson.look_for("the sliced moons and circles. k-means likes round-ish, similar-sized blobs.")
    lesson.careful("Clustering is useful, but it does not remove judgement. Shape matters.")


@lesson.step("Penguin species are hidden", beat="forreal")
def _():
    table, score = cached_penguin_clusters()
    lesson.say("Now k-means gets penguin measurements with the species labels removed.")
    st.dataframe(table, use_container_width=True)
    st.metric("cluster/species agreement", f"{score:.2f}")
    lesson.look_for("cluster numbers versus real species. Cluster 0 is not 'Adelie'; it is a number the algorithm made up.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. Find a seed that makes k-means fail on easy blobs.
2. Find the k where the elbow is clearest.
3. Quantize a photo down to 2 colours. Is it still recognisable?
"""
    )
    lesson.kid_corner("Sort laundry into two piles, then three. Which number of piles felt useful?")


lesson.finish()
