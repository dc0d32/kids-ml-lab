"""Chapter 12 · Arrows and Grids."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kidsml import lesson
from kidsml import linalg as la
from kidsml.nn_numpy import ACTIVATIONS
from kidsml.plots import (
    ACCENT,
    AMBER,
    COOL,
    EDGE,
    GHOST,
    GRIDLINE,
    PINK,
    SHAPE,
    VIOLET,
    style_plotly,
)

lesson.begin(12)

# This chapter is drawn on the same black page as everything else, so every colour here is
# one that stays readable against it. The house in particular has to be the brightest thing
# in the picture: it is the shape you watch for the rotation, the flip and the shear, and a
# dark outline on a dark panel is simply not there.
BLUE = COOL
GREEN = ACCENT
ORANGE = AMBER
PURPLE = VIOLET
INK = SHAPE
GREY = GHOST


@st.cache_data(show_spinner=False)
def cloud3d():
    return la.tilted_cloud(n=260, seed=11)


@st.cache_data(show_spinner=False)
def best_shadow_score():
    return la.best_shadow_spread(cloud3d())


def safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower())


def setup_axes(ax, limit: float = 4.0, title: str = "") -> None:
    ax.axhline(0, color=EDGE, linewidth=1.2)
    ax.axvline(0, color=EDGE, linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    if title:
        ax.set_title(title)


def draw_arrow(ax, end, *, start=(0, 0), color=BLUE, label="", linewidth=3.0) -> None:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=linewidth, color=color, shrinkA=0, shrinkB=0),
    )
    if label:
        pos = start + 0.56 * (end - start)
        ax.text(pos[0], pos[1], label, color=color, fontsize=10, weight="bold")


def draw_grid(ax, lines, *, color=BLUE, alpha=0.85, linewidth=1.0) -> None:
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color=color, alpha=alpha, linewidth=linewidth)


def matrix_sliders(prefix: str, defaults) -> np.ndarray:
    a0, b0, c0, d0 = [float(x) for x in defaults]
    a = st.slider("a: where right arrow lands on x", -3.0, 3.0, a0, 0.1, key=f"{prefix}_a")
    b = st.slider("b: where up arrow lands on x", -3.0, 3.0, b0, 0.1, key=f"{prefix}_b")
    c = st.slider("c: where right arrow lands on y", -3.0, 3.0, c0, 0.1, key=f"{prefix}_c")
    d = st.slider("d: where up arrow lands on y", -3.0, 3.0, d0, 0.1, key=f"{prefix}_d")
    return la.matrix(a, b, c, d)


def preset_matrix(prefix: str, *, index: int = 0) -> tuple[str, np.ndarray]:
    names = list(la.PRESETS)
    preset = st.selectbox("Preset", names, index=index, key=f"{prefix}_preset")
    st.caption(la.PRESET_BLURB[preset])
    key = f"{prefix}_{safe_key(preset)}"
    return preset, matrix_sliders(key, la.PRESETS[preset])


def grid_figure(M, title: str = "Matrix as a grid mover", limit: float = 2.5, step: float = 0.5):
    original = la.grid_lines(limit=limit, step=step)
    moved = la.transform_grid(M, limit=limit, step=step)
    house = la.house()
    house_moved = la.apply_matrix(house, M)
    square = la.unit_square()
    square_moved = la.apply_matrix(square, M)
    landed = la.where_the_arrows_land(M)
    basis = np.vstack([np.zeros(2), landed["right arrow (1, 0)"], landed["up arrow (0, 1)"], house_moved, square_moved])
    bound = max(3.2, float(np.max(np.abs(basis))) * 1.25)

    fig, ax = lesson.figure(6.0, 5.2)
    setup_axes(ax, bound, title)
    draw_grid(ax, original, color=GREY, alpha=0.7, linewidth=0.8)
    draw_grid(ax, moved, color=BLUE, alpha=0.95, linewidth=1.3)
    ax.fill(square_moved[:, 0], square_moved[:, 1], color=ORANGE, alpha=0.25, label="moved unit square")
    ax.plot(square_moved[:, 0], square_moved[:, 1], color=ORANGE, linewidth=2.0)
    ax.plot(house[:, 0], house[:, 1], color=GREY, linewidth=1.4, linestyle="--", alpha=0.9,
            label="house before")
    ax.plot(house_moved[:, 0], house_moved[:, 1], color=INK, linewidth=2.8, label="house after")
    draw_arrow(ax, landed["right arrow (1, 0)"], color=PINK, label="(1,0) lands here")
    draw_arrow(ax, landed["up arrow (0, 1)"], color=GREEN, label="(0,1) lands here")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.0)
    return fig


def basis_figure(M):
    landed = la.where_the_arrows_land(M)
    fig, ax = lesson.figure(5.6, 4.8)
    setup_axes(ax, 3.5, "The two starter arrows decide the whole move")
    draw_arrow(ax, [1, 0], color=GREY, label="old (1,0)", linewidth=2.0)
    draw_arrow(ax, [0, 1], color=GREY, label="old (0,1)", linewidth=2.0)
    draw_arrow(ax, landed["right arrow (1, 0)"], color=PINK, label="new (1,0)")
    draw_arrow(ax, landed["up arrow (0, 1)"], color=GREEN, label="new (0,1)")
    for point in landed.values():
        ax.scatter(point[0], point[1], s=70, color=INK, zorder=5)
    return fig


def shadow_plot(points3d, shadow, spread, best):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("3D cloud", "2D shadow"),
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter3d(
            x=points3d[:, 0], y=points3d[:, 1], z=points3d[:, 2],
            mode="markers", marker=dict(size=3, color=points3d[:, 0], colorscale="Viridis", opacity=0.75),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=shadow[:, 0], y=shadow[:, 1], mode="markers",
            marker=dict(size=6, color=shadow[:, 0], colorscale="Viridis", opacity=0.75),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="shadow width", scaleanchor="y", scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text="shadow height", row=1, col=2)
    fig.update_layout(
        title=f"Spread kept: {spread:.2f} out of best {best:.2f}",
        scene=dict(xaxis_title="3D x (left-right)", yaxis_title="3D y (front-back)", zaxis_title="3D z (up-down)"),
    )
    return style_plotly(fig, height=430)


@lesson.step("A matrix is not a box", beat="hook")
def _():
    lesson.say(
        """
You may have been told a matrix is a box of numbers. That is like being told a song is a
box of dots on lines.

Chapter 11 taught you to distrust a shiny score. Now we build the score machine from
Chapter 01 more carefully, so nothing feels like magic later when we stack these moves
into a learning machine.

Here is what it actually is: **an instruction for moving space**. A grid is the graph-paper floor: straight crossing lines, with every crossing point ready to move.

Drag the instruction, and the grid skates, flips, stretches, and wakes up.
"""
    )
    lesson.mermaid(
        """
graph TD
    A[An arrow] --> B[A grid full of arrows]
    B --> C[A matrix moves the whole grid]
    C --> D[Stacked straight moves stay straight]
    D --> E[A squish bends space into new shapes]
    E --> F[Chapter 13 gives this a name]
""",
    )
    lesson.look_for("the last two boxes. Straight moves alone buy nothing new. The bend is the trick, and Chapter 13 is where it gets a name.")


@lesson.step("The grid mover", beat="byhand")
def _():
    lesson.say(
        "This is the centre of the chapter. The four numbers `[[a, b], [c, d]]` say where the two starter arrows land: `a,c` move the right arrow and `b,d` move the up arrow. Pick a preset, then drag one number and watch the grey graph-paper floor become the blue moved floor."
    )
    knobs, picture = lesson.controls()
    with knobs:
        _, M = preset_matrix("ch12_grid", index=0)
        landed = la.where_the_arrows_land(M)
        st.write(f"`(1,0)` → **{np.round(landed['right arrow (1, 0)'], 2)}**")
        st.write(f"`(0,1)` → **{np.round(landed['up arrow (0, 1)'], 2)}**")
        st.metric("area multiplier", f"{la.area_change(M):.2f}")
    with picture:
        lesson.show(grid_figure(M, "Drag until the house does something weird"))
        lesson.look_for("the little house. Rotations, flips, and shears are harder to miss on a house than on a blob.")


@lesson.step("A vector is an arrow", beat="byhand")
def _():
    lesson.say("A vector is a little trip: **how far** and **which way**. Drag the two parts of `[3, 4]` and watch the old 3-4-5 triangle pop open.")
    knobs, picture = lesson.controls()
    with knobs:
        x = st.slider("sideways part", -6.0, 6.0, 3.0, 0.5, key="ch12_vec_x")
        y = st.slider("up-down part", -6.0, 6.0, 4.0, 0.5, key="ch12_vec_y")
        v = np.array([x, y], dtype=float)
        st.metric("length", f"{la.magnitude(v):.2f}")
        st.metric("angle", f"{la.angle_degrees(v):.1f}°")
        st.caption(f"direction arrow: {np.round(la.direction(v), 2)}")
    with picture:
        bound = max(6.0, abs(x), abs(y)) + 0.8
        fig, ax = lesson.figure(5.8, 4.8)
        setup_axes(ax, bound, "Vector = direction + length")
        ax.plot([0, x, x], [0, 0, y], color=GREY, linestyle="--", linewidth=2)
        ax.text(x / 2, -0.45, f"{x:g}", color=INK, ha="center")
        ax.text(x + 0.25, y / 2, f"{y:g}", color=INK)
        draw_arrow(ax, v, color=BLUE, label=f"[{x:g}, {y:g}]")
        lesson.show(fig)
        lesson.look_for("the dashed legs. When the parts are 3 and 4, the arrow length is 5.")


@lesson.step("Add arrows tip to tail", beat="byhand")
def _():
    lesson.say("Adding vectors means walk one arrow, then launch the next arrow from that new tip. Scaling stretches, shrinks, or flips the arrow backward.")
    knobs, picture = lesson.controls()
    with knobs:
        ax1 = st.slider("first arrow x", -4.0, 4.0, 2.0, 0.5, key="ch12_add_ax")
        ay1 = st.slider("first arrow y", -4.0, 4.0, 1.0, 0.5, key="ch12_add_ay")
        bx1 = st.slider("second arrow x", -4.0, 4.0, 1.0, 0.5, key="ch12_add_bx")
        by1 = st.slider("second arrow y", -4.0, 4.0, 2.0, 0.5, key="ch12_add_by")
        scale = st.slider("scale the second arrow", -2.0, 2.0, 1.0, 0.25, key="ch12_add_scale")
    a = np.array([ax1, ay1])
    b = scale * np.array([bx1, by1])
    total = a + b
    with picture:
        fig, ax = lesson.figure(5.8, 4.8)
        setup_axes(ax, 6.0, "Tip-to-tail addition")
        draw_arrow(ax, a, color=BLUE, label="a")
        draw_arrow(ax, a + b, start=a, color=ORANGE, label="scaled b")
        draw_arrow(ax, total, color=GREEN, label="a + scaled b", linewidth=2.4)
        lesson.show(fig)
        lesson.look_for("the green shortcut. It lands at the same place as the two-arrow walk.")
    st.metric("result", f"[{total[0]:.1f}, {total[1]:.1f}]")


@lesson.step("The dot product is agreement", beat="byhand")
def _():
    lesson.say("Two arrows can cheer each other on, ignore each other, or pull in opposite directions. The dot product is that agreement number.")
    knobs, picture = lesson.controls()
    with knobs:
        angle = st.slider("angle of the orange arrow", -180, 180, 45, 5, key="ch12_dot_angle")
        length = st.slider("orange arrow length", 0.5, 5.0, 3.0, 0.25, key="ch12_dot_len")
    watch = np.array([3.0, 0.0])
    other = length * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    score = la.dot(watch, other)
    if score > 0.2:
        verdict = "same-way energy"
    elif score < -0.2:
        verdict = "disagreement"
    else:
        verdict = "right-angle silence"
    with picture:
        fig, ax = lesson.figure(5.8, 4.8)
        setup_axes(ax, 5.2, "How much do these arrows agree?")
        draw_arrow(ax, watch, color=BLUE, label="w")
        draw_arrow(ax, other, color=ORANGE, label="x")
        lesson.show(fig)
        lesson.look_for("the orange arrow near 90°. The number falls toward zero there.")
    a, b = st.columns(2)
    a.metric("w · x", f"{score:.2f}")
    b.metric("verdict", verdict)
    lesson.aha("This is the **score machine from Chapter 01**. The weighted sum `w·x` asks one question: how much does this input point the same way as the weights `w`? Chapter 13 builds a whole learning machine on that one question.")


@lesson.step("Columns say where arrows land", beat="seeit")
def _():
    lesson.say("Read a matrix by its **columns**. Column one is the landing pad for `(1, 0)`. Column two is the landing pad for `(0, 1)`.")
    knobs, picture = lesson.controls()
    with knobs:
        M = matrix_sliders("ch12_columns", (1.0, 1.0, 0.0, 1.0))
        st.code(f"[[{M[0,0]:.1f}, {M[0,1]:.1f}],\n [{M[1,0]:.1f}, {M[1,1]:.1f}]]", language="python")
        landed = la.where_the_arrows_land(M)
        st.write(f"Right arrow lands at **{np.round(landed['right arrow (1, 0)'], 2)}**")
        st.write(f"Up arrow lands at **{np.round(landed['up arrow (0, 1)'], 2)}**")
    with picture:
        lesson.show(basis_figure(M))
        lesson.look_for("the two coloured landing arrows. Once those are known, every other arrow is forced.")
    lesson.jargon("linear", "It means grid lines stay straight and evenly spaced. No secret extra wiggle is allowed.")


@lesson.step("Area is the determinant", beat="play")
def _():
    lesson.say("The orange unit square gets grabbed and stretched into a parallelogram. Its new signed area is the determinant: the area multiplier.")
    names = list(la.PRESETS)
    picked = st.selectbox("Transformation to investigate", names, index=names.index("collapse onto a line"), key="ch12_area_preset")
    guess = lesson.predict(
        "What happens to the area with the 'collapse onto a line' preset?",
        ["It doubles", "It becomes zero", "It becomes negative"],
        correct=1,
        why="A line has zero area. The square has been pancaked into one dimension, so the determinant reads 0.",
        key="ch12_area_predict",
    )
    if guess is None:
        return
    M = la.matrix(*la.PRESETS[picked])
    lesson.show(grid_figure(M, "The determinant is the square's new signed area", limit=1.6))
    lesson.look_for("the orange shape. If it has become a line, its area is zero.")
    st.metric("determinant", f"{la.area_change(M):.2f}")
    lesson.careful("A negative determinant does not mean negative size. It means the grid flipped over, like a pancake.")


@lesson.step("Collapse throws information away", beat="play")
def _():
    lesson.say("When the determinant is 0, the whole plane lands on one line. Different points can crash into the same spot like bumper cars.")
    knobs, picture = lesson.controls()
    with knobs:
        blend = st.slider("blend toward collapse", 0.0, 1.0, 1.0, 0.05, key="ch12_collapse_blend")
        start = la.matrix(1, 0, 0, 1)
        end = la.matrix(*la.PRESETS["collapse onto a line"])
        M = (1 - blend) * start + blend * end
        st.metric("determinant", f"{la.area_change(M):.3f}")
    with picture:
        lesson.show(grid_figure(M, "A dimension disappearing", limit=1.8))
        lesson.look_for("places where many blue grid lines lie on top of each other.")
    lesson.aha("This is the preview for 3D → 2D shadows. Flattening can be useful, but lost information cannot be unflattened.")


@lesson.step("Projection is a shadow", beat="play")
def _():
    lesson.say("Projecting one arrow onto another means drop a straight shadow onto that direction. The shadow keeps only the part that agrees.")
    knobs, picture = lesson.controls()
    with knobs:
        arrow_angle = st.slider("arrow angle", -180, 180, 55, 5, key="ch12_proj_arrow_angle")
        arrow_len = st.slider("arrow length", 0.5, 5.0, 4.0, 0.25, key="ch12_proj_arrow_len")
        line_angle = st.slider("shadow line angle", -180, 180, 10, 5, key="ch12_proj_line_angle")
    v = arrow_len * np.array([np.cos(np.radians(arrow_angle)), np.sin(np.radians(arrow_angle))])
    onto = np.array([np.cos(np.radians(line_angle)), np.sin(np.radians(line_angle))])
    shadow = la.project_onto(v, onto)
    with picture:
        fig, ax = lesson.figure(5.8, 4.8)
        setup_axes(ax, 5.2, "Projection: keep the shadow")
        line = np.vstack([-5 * onto, 5 * onto])
        ax.plot(line[:, 0], line[:, 1], color=GREY, linewidth=2.0)
        draw_arrow(ax, v, color=BLUE, label="arrow")
        draw_arrow(ax, shadow, color=ORANGE, label="shadow")
        ax.plot([v[0], shadow[0]], [v[1], shadow[1]], color=GREY, linestyle="--")
        lesson.show(fig)
        lesson.look_for("the dashed drop line. The orange part is all the shadow remembers.")
    st.metric("shadow length", f"{la.magnitude(shadow):.2f}")


@lesson.step("3D to 2D: beat the shadow finder", beat="play")
def _():
    lesson.say(
        """
Here is a cloud of points floating in 3D: x is left-right, y is front-back, z is up-down. Shine a lamp on it and you get a flat 2D shadow on the wall.

Some angles give a fat, spread-out shadow. Others squash the cloud into a skinny smear, where different points land on top of each other. Spin and tilt the lamp and hunt for the widest shadow you can make.

A wide shadow keeps the differences between points. A skinny one throws them away.
"""
    )
    knobs, picture = lesson.controls()
    with knobs:
        angle_a = st.slider("spin the lamp", -180, 180, 20, 5, key="ch12_shadow_a")
        angle_b = st.slider("tilt the lamp", -80, 80, 25, 5, key="ch12_shadow_b")
    points = cloud3d()
    shadow = la.shadow_on_plane(points, angle_a, angle_b)
    spread = la.spread_of(shadow)
    best = best_shadow_score()
    with picture:
        st.plotly_chart(shadow_plot(points, shadow, spread, best), width="stretch")
        lesson.look_for("whether the 2D shadow stays wide or becomes a skinny smear.")
    a, b = st.columns(2)
    a.metric("your spread kept", f"{spread:.2f}")
    b.metric("best possible", f"{best:.2f}")
    st.progress(min(1.0, spread / best))
    lesson.aha("PCA in Chapter 21 is this search done in one step: pick the shadow that keeps the most spread.")


@lesson.step("Two grid moves can become one", beat="play")
def _():
    lesson.say("Choose two transformations. Guess whether doing A and then B can always be replaced by one combined matrix.")
    guess = lesson.predict(
        "Can a single matrix always do the same job as two linear transformations in a row?",
        ["Yes", "No", "Only for rotations"],
        correct=0,
        why="The combined matrix is `B @ A`. It lands every test point exactly where the two-step trip lands. Same footprints!",
        key="ch12_chain_predict",
    )
    if guess is None:
        return
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Matrix A**")
        A = matrix_sliders("ch12_chain_A", (1.0, 0.6, 0.0, 1.0))
    with right:
        st.markdown("**Matrix B**")
        B = matrix_sliders("ch12_chain_B", (0.7, -0.8, 0.8, 0.7))
    combined, gap = la.chained_equals_single(A, B)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    titles = ["after A", "after A then B", "after B @ A"]
    grids = [la.transform_grid(A, limit=2.0), la.transform_grid(B @ A, limit=2.0), la.transform_grid(combined, limit=2.0)]
    for ax, title, lines in zip(axes, titles, grids):
        setup_axes(ax, 4.0, title)
        draw_grid(ax, lines, color=BLUE, alpha=0.9)
    lesson.show(fig)
    lesson.look_for("the last two pictures. They match because the one matrix is the two moves fused together.")
    st.metric("largest difference on 200 test points", f"{gap:.1f}")
    st.code(f"combined =\n{np.round(combined, 2)}", language="python")


@lesson.step("Ten straight layers collapse to one", beat="play")
def _():
    lesson.say("Stacking straight moves with no squish buys no new shapes. Multiply ten of them together and you get one matrix that does the exact same job.")
    layers = st.slider("number of no-squish layers", 1, 10, 10, 1, key="ch12_linear_layers")
    base = la.matrix(1.05, 0.2, -0.1, 0.95)
    combined = np.eye(2)
    for _ in range(layers):
        combined = base @ combined
    one_step, gap = la.chained_equals_single(np.linalg.matrix_power(base, max(1, layers - 1)), base)
    st.code("layer10(layer9(...layer1(x))) == one_big_matrix @ x", language="python")
    cols = st.columns(2)
    cols[0].dataframe(pd.DataFrame(np.round(combined, 3), columns=["col 1", "col 2"]), hide_index=True)
    cols[1].metric("extra shape power", f"{gap:.1f}")
    lesson.aha("Ten straight layers with no squish do exactly what one straight layer does. Without a bend in the middle, the extra nine layers are wasted.")


@lesson.step("A squish bends the grid", beat="play")
def _():
    lesson.say(
        """
The last step proved the trap: stack straight moves with no squish, and they fuse into one
straight move. Put a squish between two matrices and the grid lines bend.

That bend is the new ingredient a single matrix cannot copy.
"""
    )
    guess = lesson.predict(
        "What do you think the squish does to the straight grid lines?",
        ["They stay straight", "They bend", "They disappear"],
        correct=1,
        why="A squish squeezes different parts of space by different amounts. The straight-line stamp bends and breaks.",
        key="ch12_squish_predict",
    )
    if guess is None:
        return
    knobs, picture = lesson.controls()
    with knobs:
        use_squish = st.toggle("put a squish in the middle", value=True, key="ch12_squish_toggle")
        names = list(ACTIVATIONS)
        activation = st.selectbox("activation", names, index=names.index("tanh"), key="ch12_squish_activation")
    A = la.matrix(0.5, -1.25, -0.25, -0.5)
    B = la.matrix(1.25, 1.5, 0.0, -1.5)
    squish = activation if use_squish else None
    lines = la.transform_grid_with_squish(A, B, squish=squish, limit=3.0, step=0.5)
    gap = la.squish_breaks_the_pattern(A, B, activation) if use_squish else la.chained_equals_single(A, B)[1]
    with picture:
        fig, ax = lesson.figure(6.2, 5.0)
        setup_axes(ax, 4.6, "Straight lines versus bent lines")
        draw_grid(ax, lines, color=PURPLE if use_squish else BLUE, alpha=0.9, linewidth=1.2)
        lesson.show(fig)
        lesson.look_for("the grid lines. Toggle the squish off and they snap straight again.")
    st.metric("difference from one plain matrix", f"{gap:.2f}")
    lesson.aha("That bend is the whole point of the next chapters. Chapter 13 puts one squish on one score machine and gives it a name. Chapter 15 stacks many of them so a hidden layer can invent brand-new features.")


@lesson.step("For real: does stacking buy anything?", beat="forreal")
def _():
    lesson.say("The weighted sum `z = w1*x1 + w2*x2 + b` is the score machine from Chapter 01, and it is the same arrow-agreement you have been dragging all chapter. Chapter 13 gives it a name. Below, two matrix moves run with and without a squish between them.")
    knobs, picture = lesson.controls()
    with knobs:
        x1 = st.slider("input x1", -3.0, 3.0, 1.0, 0.25, key="ch12_real_x1")
        x2 = st.slider("input x2", -3.0, 3.0, -1.0, 0.25, key="ch12_real_x2")
    with picture:
        st.code(
            """
x = np.array([x1, x2])
W1 = np.array([[1.0, 2.0], [-0.5, 1.0]])
W2 = np.array([[2.0, -1.0], [0.5, 1.5]])
b = np.array([0.25, -0.75])

z = W1 @ x + b
linear_gap = max(abs(W2 @ (W1 @ x) - (W2 @ W1) @ x))
squish_gap = max(abs(W2 @ np.tanh(W1 @ x) - (W2 @ W1) @ x))
""",
            language="python",
        )
    x = np.array([x1, x2])
    W1 = np.array([[1.0, 2.0], [-0.5, 1.0]])
    W2 = np.array([[2.0, -1.0], [0.5, 1.5]])
    b = np.array([0.25, -0.75])
    z = W1 @ x + b
    linear_gap = float(np.max(np.abs(W2 @ (W1 @ x) - (W2 @ W1) @ x)))
    squish_gap = float(np.max(np.abs(W2 @ np.tanh(W1 @ x) - (W2 @ W1) @ x)))
    table = pd.DataFrame(
        {
            "quantity": ["z = W1 @ x + b", "no-squish gap", "with tanh gap"],
            "value": [str(np.round(z, 3)), f"{linear_gap:.8f}", f"{squish_gap:.3f}"],
        }
    )
    st.dataframe(table, hide_index=True, width="content")
    lesson.look_for("the two gap rows. Linear stacking matches exactly; tanh breaks the match.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go bend space", beat="challenge")
def _():
    lesson.say(
        """
1. **Rotate and double.** Find a matrix that spins the house and makes it twice as big.
2. **Undo button.** Find two different matrices whose product is the identity, so the second one undoes the first.
3. **Upside-down house.** Make the house land on its roof.
4. **Worst shadow.** In the 3D shadow game, hunt for the smallest spread.
5. **Area bet.** Predict the determinant before checking the area meter.
6. **Bonus:** hunt for arrows that keep their direction under a transformation. Grown-ups call those **eigenvectors**.
"""
    )
    lesson.kid_corner(
        "Use a torch and your hand to make shadows on a wall. Stretch a drawing on a balloon. Squash a photo on a screen. Same idea: the object stays one thing, but the space around it gets moved."
    )


lesson.finish()
