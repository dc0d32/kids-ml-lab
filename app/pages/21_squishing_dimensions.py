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
graph TD
    A[High-dimensional cloud] --> B[Choose an angle]
    B --> C[Cast a shadow]
    C --> D[2D plot]
    D --> E[Check what stayed spread out]
""",
    )
    lesson.look_for("the last box. Spread means the points stayed far apart after the squish.")


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
        key="ch21_hand_shadow",
    )
    if guess is None:
        return
    a, b = st.columns(2)
    a.metric("x spread", f"{spread(table['x shadow']):.1f}")
    b.metric("y spread", f"{spread(table['y shadow']):.1f}")
    lesson.aha("Spread is information because it keeps points different from each other.")
    lesson.jargon("variance", "Spread measured from the middle. A bigger variance means the shadow kept the points pulled farther apart.")
    lesson.jargon("projection", "Casting data onto a smaller shadow, like keeping x but dropping y.")
    lesson.jargon("principal component", "One best-shadow direction. PCA keeps the first few of these directions.")
    lesson.jargon("principal component analysis", "PCA: search for the shadow where the points stay as spread out as possible.")


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
        key="ch21_pca_angle",
    )
    if guess is None:
        return

    X = cached_cloud()
    knobs, picture = lesson.controls()
    with knobs:
        yaw = st.slider("shadow angle left-right", -90, 90, 0, 5, key="ch21_yaw")
        pitch = st.slider("shadow tilt", -80, 80, 0, 5, key="ch21_pitch")
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
        key="ch21_digit_components",
    )
    if guess is None:
        return
    knobs, picture = lesson.controls()
    with knobs:
        index = st.slider("which digit image?", 0, 50, 8, key="ch21_digit_index")
        components = st.slider("components kept", 1, 64, 12, key="ch21_components")
    fig, curve = plot_reconstruction(index=index, n_components=components)
    with picture:
        lesson.show(fig)
        lesson.look_for("the rebuilt digit. Drag until your eyes stop trusting it.")
    lesson.show(plot_variance_curve(curve, components))
    lesson.look_for("where your component count hits the curve. The first few directions do most of the work!")
    lesson.say("Those rebuilt digits are mixed from ghostly directions. PCA did not store a tiny 8; it stored directions that can rebuild many digits.")
    lesson.show(plot_eigendigits(8))
    lesson.look_for("strokes that look like pieces of many digits at once, not one clean number.")


@lesson.step("Two maps of hidden digits", beat="forreal")
def _():
    lesson.say(
        """
Now hide every digit label, squish the pixel numbers to two PCA shadows, and colour the labels after the fact.

Clusters appear even though PCA never saw a digit label. Surprise: the shadow kept enough shape to reveal islands!
"""
    )
    fig, kept = cached_digits_pca_plot()
    left, right = st.columns(2, gap="large")
    with left:
        lesson.show(fig)
        st.caption(f"PCA kept {kept:.1%} of the pixel spread in two shadows.")
    with right:
        lesson.show(cached_tsne_plot())
        st.caption("t-SNE is not PCA. It keeps nearby digits near each other, then bends the map to show neighbourhoods.")
    lesson.look_for("digits that overlap in PCA, then islands in t-SNE. Shared strokes make shared mistakes.")
    st.metric("variance kept in two PCA shadows", f"{kept:.1%}")
    lesson.careful(
        "t-SNE is a different method from PCA. It bends and stretches the map to make local neighbourhoods visible, so the distances can lie: the gap between islands, or the island size, is not a measured fact."
    )


@lesson.step("Where the flat shadow gives up", beat="forreal")
def _():
    lesson.say(
        """
You just watched PCA find great shadows. Now here is the one thing it cannot do.

PCA only picks **flat** angles — straight-line directions through the cloud. So what shape
breaks that? Two rings, one tucked inside the other, like a small hoop sitting inside a big
hoop.
"""
    )
    lesson.say(
        """
The only thing that tells the rings apart is **distance from the middle**: inner dots sit
about 0.4 out, outer dots about 1.0 out. "Distance from the middle" is a curved idea, not a
straight direction. We even handed PCA a third number for every dot — its exact distance
from the middle, the perfect clue. But PCA keeps the directions with the most spread, and
the widest spread is the flat pancake of the rings. So it keeps the pancake and throws the
distance number away.
"""
    )
    left, right = st.columns(2, gap="large")
    with left:
        fig = plot_pca_linear_failure()
        fig.axes[0].set_title("Two rings, one flat shadow")
        lesson.show(fig)
        lesson.look_for("the two colours sitting right on top of each other. Untangling the rings needed a bend, and PCA does not bend.")
    loadings, first_kept = cached_penguin_pca()
    with right:
        st.markdown("**Penguin first shadow**")
        st.dataframe(loadings, hide_index=True, width="stretch")
        lesson.look_for("which measurements get the biggest weights. Body-mass and flipper-length lead, so shadow 1 mostly means how big the penguin is.")
    st.metric("penguin first-component variance", f"{first_kept:.1%}")
    lesson.say(
        """
Back to an easy shape. On a plain table of penguin measurements the same squishing works
fine, and the first shadow reads mostly like body size.

So the lesson is the ceiling: PCA is a flat tool. When two classes are wrapped around each
other, no flat angle can separate them, and you need a tool that can curve — a kernel, or a
neural network.
"""
    )


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
