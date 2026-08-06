"""Chapter 20 · Sorting Without Labels."""

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

lesson.begin(20)


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
Every model so far was shown the right answers. Somebody had to label the data.

Part 5 asks a different question: **what can you learn when nobody tells you the answers?**

Nobody labels anything this time. Here is a pile of dots. Find the clumps. You already do
this when you sort laundry into piles before anyone hands you the pile names.
"""
    )
    lesson.careful(
        "The danger is that 'clump' sounds like a thing everyone will agree on. Sometimes one person's two piles are another person's three piles. The dots do not wear name tags."
    )
    lesson.show(plot_kmeans_stage(kmeans_history(k=3, seed=0, bad_start=False)[0]))
    lesson.look_for("the dots before they have names. Your eyes start hunting for piles before the algorithm says a word.")
    lesson.kid_corner("Put blocks on the floor. Move them into piles by what feels near. You do not need names for the piles before you start.")
    lesson.jargon("clustering", "Sorting data into groups when the answer labels are missing.")
    lesson.jargon("cluster", "One of those groups: points that ended up in the same pile.")


@lesson.step("Two steps you can recite", beat="byhand")
def _():
    lesson.say("**Step 1:** every point joins the nearest centre. **Step 2:** every centre scoots to the middle of its members. Repeat until nothing changes.")
    lesson.mermaid(
        """
graph TD
    A[Choose centres] --> B[Assign points]
    B --> C[Move centres]
    C --> D{Anything changed?}
    D -->|yes| B
    D -->|no| E[Stop]
""",
    )
    lesson.look_for("the loop: assign, move, check. The same two-step dance repeats until the centres stop moving.")
    lesson.say("Why must it stop? Each round either makes total squared point-to-centre distance smaller, or nothing changes. There are only so many possible assignments, so the loop cannot shrink forever.")


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
        st.dataframe(assignments, hide_index=True, width="stretch")
        st.markdown("**Move centres**")
        st.dataframe(new_centres, hide_index=True, width="stretch")
    lesson.look_for("which centre each point joins, then where the new centre lands after the averages pull it.")
    lesson.say("Left moves to `((1+1+2)/3, (1+2+1)/3) = (1.33, 1.33)`. Right moves to `((7+8+7)/3, (7+7+8)/3) = (7.33, 7.33)`.")
    lesson.aha("Round two changes nothing. That means the algorithm has converged: it stopped moving!")
    lesson.jargon("k-means", "Pick k centres, assign points to nearest centres, move centres to group middles, and repeat.")
    lesson.jargon("centroid", "The middle of a cluster. k-means centres move to the centroid of their points.")


@lesson.step("Press the step button", beat="seeit")
def _():
    k = st.slider("How many centres?", 2, 5, 3, key="ch20_k")
    seed = st.slider("Starting-position seed", 0, 12, 0, key="ch20_seed")
    bad = st.checkbox("Start two centres inside the same clump", value=False, key="ch20_bad")
    settings = (k, seed, bad)
    if st.session_state.get("ch20_settings") != settings:
        st.session_state["ch20_settings"] = settings
        st.session_state["ch20_step"] = 0
    history = kmeans_history(k=k, seed=seed, bad_start=bad)
    if st.button("One k-means step ▶", key="ch20_step_button"):
        st.session_state["ch20_step"] = (st.session_state.get("ch20_step", 0) + 1) % len(history)
    step = st.session_state.get("ch20_step", 0)
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
        why="k-means can only improve from its starting guess. A bad start can settle into a bad but stable setup.",
        key="ch20_bad_start",
    )
    if guess is None:
        return
    history = kmeans_history(k=3, seed=0, bad_start=True)
    step = st.slider("Bad-start stage", 0, len(history) - 1, len(history) - 1, key="ch20_bad_stage")
    fig = plot_kmeans_stage(history[step])
    lesson.show(fig)
    lesson.look_for("the clump that shares or misses a centre. A bad start can settle into that mistake and never fix itself.")
    lesson.say("That explains sklearn's defaults. k-means++ spreads the starting centres out on purpose, then sklearn tries several starts and keeps the best result.")


@lesson.step("How many clumps?", beat="play")
def _():
    lesson.say("An elbow plot uses **inertia**: add up each point's squared distance to its own centre. Smaller means tighter piles.")
    left, right = st.columns(2, gap="large")
    with left:
        lesson.show(plot_elbow("obvious"))
    with right:
        lesson.show(plot_elbow("ambiguous"))
    lesson.look_for("where the line stops dropping fast. The sharp plot argues for k = 3; the mushy one asks for judgement.")
    lesson.say("Inertia always falls as k rises because adding a centre gives the algorithm another bucket for points.")
    lesson.careful("With one centre per point, inertia hits zero and the clusters teach you nothing. The elbow asks where the extra pile stops earning its keep.")


@lesson.step("Predict five colours", beat="play")
def _():
    lesson.say(
        "A photo is a grid of pixels, but k-means sees a pixel as a **point in 3D colour space**: "
        "`(red amount, green amount, blue amount)`. Five colours means five centres in that colour space, then every pixel gets repainted by its nearest centre."
    )
    guess = lesson.predict(
        "If a photo is repainted with only five colours, what do you expect?",
        ["It stays recognisable but poster-like", "It becomes random noise", "It becomes a perfect copy"],
        correct=0,
        why="k-means chooses a small palette, then repaints every pixel with its nearest palette colour.",
        key="ch20_five_colours",
    )
    if guess is None:
        return
    uploaded = st.file_uploader("Upload a photo if you want. If not, we use sklearn's flower.", type=["png", "jpg", "jpeg"], key="ch20_upload")
    image = uploaded_image_to_array(uploaded) if uploaded is not None else cached_default_image()
    colour_k = st.slider("palette size", 2, 16, 5, key="ch20_palette")
    rebuilt, palette = cached_quantized(image, colour_k)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.image(image, caption="original", width="stretch")
    with col_b:
        st.image(rebuilt, caption=f"{colour_k}-colour version", width="stretch")
    lesson.look_for("which colours survived in the rebuilt photo. The image keeps its shape while many tiny colour differences get squeezed out.")
    lesson.show(plot_palette(palette))
    lesson.look_for("the palette swatches. k-means found these colours from sampled pixels, then repainted every pixel.")


@lesson.step("Predict the moon failure", beat="forreal")
def _():
    guess = lesson.predict(
        "Will k-means separate two crescent moons cleanly?",
        ["Yes, because there are two groups", "No, because centres make round-ish chunks", "Only if k is 10"],
        correct=1,
        why="Each centre owns nearby points, making straight-ish borders. Crescents need a curved border.",
        key="ch20_moons_fail",
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
    lesson.say("Now k-means gets penguin measurements with the species labels peeled off.")
    st.dataframe(table, width="stretch")
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
2. **Elbow hunt.** Find the k where the elbow is clearest.
3. Quantize a photo down to 2 colours. Is it still recognisable?
"""
    )
    lesson.kid_corner("Sort laundry into two piles, then three. Which number of piles felt useful?")


lesson.finish()
