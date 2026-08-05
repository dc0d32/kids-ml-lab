"""Chapter 21 · Squishing Dimensions."""

from __future__ import annotations

import streamlit as st

from kidsml import lesson
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

lesson.begin(21)


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


@lesson.step("Pick the best shadow", beat="hook")
def _():
    lesson.say(
        """
Hold up your hand and cast a shadow on a wall.

Turn it. Some shadows still shout hand. Other shadows become a pancake blob. Same hand.
Same wall. Different angle. **Choosing the angle is the whole of PCA.**
"""
    )
    lesson.kid_corner("Use a flashlight and your hand. Turn your hand until the shadow tells the best story.")
    lesson.mermaid(
        """
graph LR
    A[High-dimensional cloud] --> B[Choose an angle]
    B --> C[Cast a shadow]
    C --> D[2D plot]
    D --> E[Check what stayed spread out]
""",
        height=240,
    )
    lesson.look_for("the last box. PCA judges a shadow by how much spread survives the squish.")


@lesson.step("Keep one number", beat="byhand")
def _():
    lesson.say(
        """
These four points lie nearly sideways. Project them onto x, then onto y.

If two points land on the same shadow spot, the shadow forgot the difference between them.
If the points stay spread apart, the shadow kept more information about who is who.
"""
    )
    table = pca_hand_table()
    st.dataframe(table, hide_index=True, width="stretch")
    guess = lesson.predict(
        "Which one-number shadow keeps more of the story?",
        ["The x shadow", "The y shadow", "They keep the same amount"],
        correct=0,
        why="The x shadow keeps A and D far apart, while the y shadow squishes pairs together.",
        key="ch20_hand_shadow",
    )
    if guess is None:
        return
    a, b = st.columns(2)
    a.metric("x spread", f"{spread(table['x shadow']):.1f}")
    b.metric("y spread", f"{spread(table['y shadow']):.1f}")
    lesson.aha("Spread is information because it keeps points different from each other.")
    lesson.jargon("principal component analysis", "Search for the shadow where the points stay as spread out as possible.")


@lesson.step("Rotate the cloud", beat="seeit")
def _():
    lesson.say(
        """
Now try the same idea in 3D. Your job is to rotate the cloud and keep the widest shadow.

A tight blob means the shadow forgot most of the differences between the points.
"""
    )
    guess = lesson.predict(
        "Which angle do you expect PCA to choose?",
        ["The angle with the widest shadow", "The angle with the neatest circle", "The angle with the smallest blob"],
        correct=0,
        why="PCA does not care about prettiness. It keeps the shadow with the most spread.",
        key="ch20_pca_angle",
    )
    if guess is None:
        return

    X = cached_cloud()
    knobs, picture = lesson.controls()
    with knobs:
        yaw = st.slider("shadow angle left-right", -90, 90, 0, 5, key="ch20_yaw")
        pitch = st.slider("shadow tilt", -80, 80, 0, 5, key="ch20_pitch")
    shadow = shadow_projection(X, yaw, pitch)
    _, pca_keep = pca_shadow_answer(X)
    your_keep = variance_captured(X, shadow)
    with picture:
        st.plotly_chart(plot_shadow_3d(X), width="stretch")
        lesson.look_for("the long direction in the 3D cloud. A good shadow does not point straight along it.")
        lesson.show(plot_shadow_2d(shadow, "Your shadow"))
        lesson.look_for("whether the points stay wide apart or collapse into a tiny smudge.")
    a, b = st.columns(2)
    a.metric("your variance kept", f"{your_keep:.1%}")
    b.metric("PCA variance kept", f"{pca_keep:.1%}")


@lesson.step("When does an 8 stop being an 8?", beat="play")
def _():
    lesson.say("A digit image has 64 pixel numbers. PCA rebuilds it after keeping only the loudest directions.")
    guess = lesson.predict(
        "How many components do you think an 8 needs before you can read it?",
        ["1 or 2", "Around 8 to 12", "Nearly all 64"],
        correct=1,
        why="Most digits stay readable after PCA keeps a surprisingly small number of directions.",
        key="ch20_digit_components",
    )
    if guess is None:
        return
    knobs, picture = lesson.controls()
    with knobs:
        index = st.slider("which digit image?", 0, 50, 8, key="ch20_digit_index")
        components = st.slider("components kept", 1, 64, 12, key="ch20_components")
    fig, curve = plot_reconstruction(index=index, n_components=components)
    with picture:
        lesson.show(fig)
        lesson.look_for("the rebuilt digit. Drag until your eyes stop trusting it.")
    lesson.show(plot_variance_curve(curve, components))
    lesson.look_for("where your component count hits the curve. The first few directions do most of the work!")


@lesson.step("The ghost digits", beat="play")
def _():
    lesson.say("Those rebuilt digits are mixed from ghostly directions. PCA did not store a tiny 8; it stored directions that can rebuild many digits.")
    lesson.show(plot_eigendigits(8))
    lesson.look_for("strokes that look like pieces of many digits at once, not one clean number.")


@lesson.step("Labels come after the squish", beat="forreal")
def _():
    lesson.say(
        """
Now hide every digit label, squish the pixel numbers to two PCA shadows, and colour the labels after the fact.

Clusters appear even though PCA never saw a digit label. Surprise: the shadow kept enough shape to reveal islands!
"""
    )
    fig, kept = cached_digits_pca_plot()
    lesson.show(fig)
    lesson.look_for("digits that overlap. Those are often the same pairs that confused Chapter 17.")
    st.metric("variance kept in two PCA shadows", f"{kept:.1%}")


@lesson.step("A prettier map can lie", beat="forreal")
def _():
    lesson.say("t-SNE keeps neighbours together, so the picture can look cleaner than PCA.")
    lesson.show(cached_tsne_plot())
    lesson.look_for("islands of nearby digits, not the exact size of gaps between islands.")
    lesson.careful(
        "t-SNE bends and stretches the map to make local neighbourhoods visible. The gap between two islands, or the size of an island, is not a measured fact."
    )


@lesson.step("Flat shadows have limits", beat="forreal")
def _():
    lesson.say("PCA is linear. It can pick a flat shadow, not unwrap every shape.")
    lesson.show(plot_pca_linear_failure())
    lesson.look_for("the two curved arms crossing in the shadow. No flat angle can untangle them cleanly.")
    lesson.careful("If two curved arms cross in every flat shadow, PCA cannot separate them no matter which angle it chooses.")


@lesson.step("Penguins in one direction", beat="forreal")
def _():
    lesson.say("The same squishing idea works on ordinary tables too. Here PCA picks a first shadow through penguin measurements.")
    loadings, first_kept = cached_penguin_pca()
    st.dataframe(loadings, hide_index=True, width="stretch")
    st.metric("penguin first-component variance", f"{first_kept:.1%}")
    lesson.look_for("body-mass and flipper-length weights. Big weights often read like a size direction for penguins.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go make shadows", beat="challenge")
def _():
    lesson.say(
        """
1. How many components before you can still read a digit?
2. Which two digits stay tangled longest in the PCA plot?
3. Run PCA on penguins and decide whether the first shadow is mostly penguin size.
4. 🧸 **Little Kid Corner side quest:** Make hand shadows. Which angle tells the best story?
"""
    )


lesson.finish()
