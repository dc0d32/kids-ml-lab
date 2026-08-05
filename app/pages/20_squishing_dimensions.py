"""Chapter 20 · Squishing Dimensions."""

from __future__ import annotations

import streamlit as st

from kidsml import ui
from kidsml.unsupervised import (
    make_shadow_cloud,
    pca_hand_table,
    pca_shadow_answer,
    penguin_pca_table,
    plot_digits_pca,
    plot_digits_tsne,
    plot_eigendigits,
    plot_pca_linear_failure,
    plot_reconstruction,
    plot_shadow_2d,
    plot_shadow_3d,
    plot_variance_curve,
    shadow_projection,
    spread,
    variance_captured,
)

ui.page_setup(20)


@st.cache_data(show_spinner=False)
def cached_cloud():
    return make_shadow_cloud()


@st.cache_data(show_spinner=False)
def cached_digits_pca_plot():
    return plot_digits_pca()


@st.cache_data(show_spinner=False)
def cached_tsne_plot():
    return plot_digits_tsne(n=600, seed=0)


@st.cache_data(show_spinner=False)
def cached_penguin_pca():
    return penguin_pca_table()


# ---------------------------------------------------------------------------
ui.beat("hook", "Pick the best shadow.")

st.markdown(
    """
Hold up your hand and cast a shadow on a wall.

Turn it. Some shadows still tell you it is a hand. Other shadows become a blob.
Same hand. Same wall. Different angle.

**Choosing the angle is the whole of PCA.**

The goal is not to keep every fact. Squishing always throws something away. The goal is to
throw away the quiet directions first and keep the directions where points still look
different from each other.
"""
)

ui.little_kid_corner("Use a flashlight and your hand. Turn your hand until the shadow tells the best story.")

ui.mermaid(
    """
graph LR
    A[High-dimensional cloud] --> B[Choose an angle]
    B --> C[Cast a shadow]
    C --> D[2D plot]
    D --> E[Check what stayed spread out]
""",
    height=240,
)

st.markdown(
    "Look at the last box. PCA judges a shadow by how much spread remains after the squish."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Keep one number.")

st.markdown(
    """
These four points lie nearly sideways. Project them onto x, then onto y.

"Keep the points spread out" sounds visual, but it means something practical. If two
points land on the same shadow spot, the shadow forgot the difference between them. If the
points stay spread apart, the shadow kept more information about who is who.
"""
)
table = pca_hand_table()
st.dataframe(table, hide_index=True, use_container_width=True)
st.write("x spread:", spread(table["x shadow"]))
st.write("y spread:", spread(table["y shadow"]))
ui.aha(
    "The x shadow stays more spread out. It keeps A and D far apart, while the y shadow "
    "squishes pairs together. Spread is information because it keeps points different from each other."
)
ui.jargon("principal component analysis", "Search for the shadow where the points stay as spread out as possible.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Try to beat the algorithm.")

X = cached_cloud()
yaw = st.slider("shadow angle left-right", -90, 90, 0, 5, key="ch20_yaw")
pitch = st.slider("shadow tilt", -80, 80, 0, 5, key="ch20_pitch")
shadow = shadow_projection(X, yaw, pitch)
pca_shadow, pca_keep = pca_shadow_answer(X)
your_keep = variance_captured(X, shadow)
col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.plotly_chart(plot_shadow_3d(X), use_container_width=True)
with col_b:
    ui.show(plot_shadow_2d(shadow, "Your shadow"))
st.metric("your variance kept", f"{your_keep:.1%}")
st.metric("PCA variance kept", f"{pca_keep:.1%}")
st.caption(
    "If the shadow collapses to one tight spot, it has forgotten how the points differ. A wider shadow kept more of the story."
)

# ---------------------------------------------------------------------------
ui.beat("play", "Compress a digit and rebuild it.")

index = st.slider("which digit image?", 0, 50, 8, key="ch20_digit_index")
components = st.slider("components kept", 1, 64, 12, key="ch20_components")
fig, curve = plot_reconstruction(index=index, n_components=components)
ui.show(fig)
ui.show(plot_variance_curve(curve, components))
ui.show(plot_eigendigits(8))
st.caption("Look at the rebuild first, then the curve. Those ghostly pictures are the directions PCA mixes to rebuild digits.")

# ---------------------------------------------------------------------------
ui.beat("forreal", "Digits, labels hidden until the colouring step.")

fig, kept = cached_digits_pca_plot()
ui.show(fig)
st.metric("variance kept in two PCA shadows", f"{kept:.1%}")
st.markdown(
    """
Clusters appear even though PCA never saw a digit label. The labels only colour the picture
after the squishing is done.

Compare this with Chapter 16's confusion matrix. Digits that tangled there often sit near
or on top of each other here too. Two different tools are agreeing about which digits look alike.
"""
)

ui.show(cached_tsne_plot())
ui.careful(
    "t-SNE keeps neighbours together, so the picture can look cleaner. But it bends and "
    "stretches the map to make local neighbourhoods visible. The gap between two islands, "
    "or the size of an island, is not a measured fact. People over-read these plots because they look like maps."
)

ui.show(plot_pca_linear_failure())
ui.careful(
    "PCA is linear. It can pick a flat shadow, not unwrap every shape. If two curved arms "
    "cross in every flat shadow, PCA cannot separate them no matter which angle it chooses."
)

loadings, first_kept = cached_penguin_pca()
st.dataframe(loadings, hide_index=True, use_container_width=True)
st.metric("penguin first-component variance", f"{first_kept:.1%}")
st.caption("Look for body-mass and flipper-length weights. Big positive and negative weights often read like a size direction for penguins.")

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. How many components before you can still read a digit?
2. Which two digits stay tangled longest in the PCA plot?
3. Run PCA on penguins and decide whether the first shadow is mostly penguin size.
4. 🧸 **Little Kid Corner:** Make hand shadows. Which angle tells the best story?
"""
)

ui.worksheet_link(20)
