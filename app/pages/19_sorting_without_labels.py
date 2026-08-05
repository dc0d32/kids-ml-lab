"""Chapter 19 · Sorting Without Labels."""

from __future__ import annotations

import streamlit as st

from kidsml import ui
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

ui.page_setup(19)


@st.cache_data(show_spinner=False)
def cached_default_image():
    return default_flower_image()


@st.cache_data(show_spinner=False)
def cached_quantized(image, k):
    return quantize_image(image, k=k)


@st.cache_data(show_spinner=False)
def cached_penguin_clusters():
    return penguin_kmeans_table()


# ---------------------------------------------------------------------------
ui.beat("hook", "Find the piles without labels.")

st.markdown(
    """
Nobody labels anything this time.

Here is a pile of dots. Find the clumps.

You already do this. You sort laundry into piles without anyone telling you the pile names.
Shirts here. Socks there. Mystery hoodie in the middle.
"""
)

ui.little_kid_corner(
    "Put blocks on the floor. Move them into piles by what feels near. You do not need names for the piles before you start."
)
ui.jargon("clustering", "Sorting data into groups when the answer labels are missing.")

# ---------------------------------------------------------------------------
ui.beat("byhand", "Two steps you can recite.")

st.markdown("**Step 1:** every point joins the nearest centre. **Step 2:** every centre moves to the middle of its members. Repeat until nothing changes.")
X, centres = kmeans_hand_points()
st.write("Six points:", X)
st.write("Starting centres:", centres)
assignments, new_centres = kmeans_hand_round()
st.dataframe(assignments, hide_index=True, use_container_width=True)
st.dataframe(new_centres, hide_index=True, use_container_width=True)
ui.aha("Round two changes nothing. That means the algorithm has converged: it stopped moving.")
ui.jargon("k-means", "Pick k centres, assign points to nearest centres, move centres to group middles, and repeat.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Press the step button.")

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
ui.show(fig)
st.caption(history[step]["caption"])
st.caption("Bad starts can get stuck. That is why libraries use k-means++ starts and run several tries, keeping the best one.")

# ---------------------------------------------------------------------------
ui.beat("play", "How many clumps should there be?")

left, right = st.columns(2, gap="large")
with left:
    ui.show(plot_elbow("obvious"))
with right:
    ui.show(plot_elbow("ambiguous"))
st.markdown("Inertia means total distance from every point to its own centre. It always goes down as k rises. With one centre per point, it hits zero and teaches you nothing.")
ui.careful("The elbow is a judgement call. Sometimes it is sharp. Sometimes it is mashed potato.")

st.markdown("### Squeeze a photo to a few colours")
uploaded = st.file_uploader("Upload a photo if you want. If not, we use sklearn's flower.", type=["png", "jpg", "jpeg"])
image = uploaded_image_to_array(uploaded) if uploaded is not None else cached_default_image()
colour_k = st.slider("palette size", 2, 16, 5, key="ch19_palette")
rebuilt, palette = cached_quantized(image, colour_k)
col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.image(image, caption="original", use_container_width=True)
with col_b:
    st.image(rebuilt, caption=f"{colour_k}-colour version", use_container_width=True)
ui.show(plot_palette(palette))
st.caption("We fit on up to 5000 sampled pixels, then repaint every pixel. That little shortcut keeps it fast.")

# ---------------------------------------------------------------------------
ui.beat("forreal", "Penguin species, hidden from the model.")

left, right, third = st.columns(3, gap="large")
with left:
    ui.show(plot_kmeans_failure("moons"))
with right:
    ui.show(plot_kmeans_failure("circles"))
with third:
    ui.show(plot_dbscan_moons())
ui.careful("k-means likes round-ish, similar-sized blobs. It is not a magic 'find all groups' button.")

table, score = cached_penguin_clusters()
st.markdown("Now k-means gets penguin measurements with the species labels removed.")
st.dataframe(table, use_container_width=True)
st.metric("cluster/species agreement", f"{score:.2f}")
st.caption("Cluster 0 is not 'Adelie'. It is a number the algorithm made up. The cluster names are arbitrary.")

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find a seed that makes k-means fail on easy blobs.
2. Find the k where the elbow is clearest.
3. Quantize a photo down to 2 colours. Is it still recognisable?
4. 🧸 **Little Kid Corner:** Sort laundry into two piles, then three. Which number of piles felt useful?
"""
)

ui.worksheet_link(19)
