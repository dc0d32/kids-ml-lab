"""Two clips where the motion is the whole point.

Chapter 3 and Chapter 7 both make the same kind of claim — *give the data one more
dimension, or one more inch of room, and a problem a straight line could never touch
falls apart* — and both, until now, asked the reader to picture the motion from a still
frame. A frozen 3D scatter you have to rotate in your head is exactly the moment a nervous
reader gives up.

So we animate the two moments that carry the chapters:

* ``lift_gif_bytes`` — the ring-and-middle circle data starts perfectly flat (frame one is
  the 2D scatter the reader has been staring at), then every point rises to its own height
  ``x3 = x1² + x2²``. The ring climbs into a bowl rim, the middle stays on the floor, and a
  flat plane slides through the gap that opens up. The camera starts looking straight down —
  so the third dimension is something the reader watches appear, not something asserted.

* ``road_gif_bytes`` — a separating line that hugs one class grows a margin outward until
  both edges jam against the closest points, which light up as the support vectors. The
  moment the widening *stops* because it hit something is the lesson.

Both render straight to a looping GIF in memory with :mod:`kidsml.anim`, so nothing new is
installed and nothing touches the disk. Cache them on a page with ``@st.cache_data`` and
show with ``st.image``; in a notebook use ``IPython.display.Image(data=...)``.
"""

from __future__ import annotations

import numpy as np

from kidsml import anim
from kidsml.datasets import toy_shape
from kidsml.plots import ACCENT, AMBER, COOL, EDGE, GRIDLINE, INK, WARM, use_house_style

# The two points the chapter works out by hand: a ring point that climbs to the top and a
# middle point that barely leaves the floor. Showing both makes the split obvious.
_CALLOUTS = np.array([[2.0, 0.0], [0.3, 0.4]])
_CALLOUT_LABELS = ("(2, 0):  2² + 0² = 4", "(0.3, 0.4):  0.3² + 0.4² = 0.25")

# A flat sheet at this height sits cleanly in the gap: middle points top out near 0.85, ring
# points bottom out near 2.96, so nothing is anywhere close to it.
_PLANE_Z = 1.8


def _style_3d(ax) -> None:
    """Make a 3D axes match the pure-black house page."""
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        axis.pane.set_edgecolor(EDGE)
        axis._axinfo["grid"]["color"] = GRIDLINE
        axis.line.set_color(EDGE)
    ax.tick_params(colors="#B0B0B9", labelsize=6)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.zaxis.label.set_color(INK)


def lift_gif_bytes(n: int = 130, seed: int = 1, fps: int = 12) -> bytes:
    """The circle data lifting into 3D, then a flat plane sliding through the gap.

    The clip runs in three moves the reader can name: the points sit flat under a nearly
    top-down camera (so it *is* the 2D picture), they rise to ``x3 = x1² + x2²`` while the
    camera tilts side-on, and a flat plane slides in between the risen ring and the low
    middle. The two by-hand points from the chapter, ``(2, 0)`` and ``(0.3, 0.4)``, are
    called out as they move so the arithmetic and the motion agree.
    """
    use_house_style()

    import matplotlib.pyplot as plt

    X, y = toy_shape("circles", n=n, noise=0.08, seed=seed)
    height = X[:, 0] ** 2 + X[:, 1] ** 2
    colours = np.where(y == 1, WARM, COOL)
    call_h = _CALLOUTS[:, 0] ** 2 + _CALLOUTS[:, 1] ** 2

    fig = plt.figure(figsize=(4.7, 4.6))
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax)

    scat = ax.scatter(X[:, 0], X[:, 1], np.zeros(n), c=colours, s=22, depthshade=True)
    hi = ax.scatter(
        _CALLOUTS[:, 0], _CALLOUTS[:, 1], np.zeros(len(_CALLOUTS)),
        c=[AMBER, AMBER], s=90, edgecolor=INK, linewidth=1.2, depthshade=False,
    )

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_zlim(0, 4.4)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # Held back until the camera has tilted. Looking straight down there is no visible
    # z-axis for it to label, so it lands in the corner looking like a mistake.
    ax.set_zlabel("")
    ax.set_box_aspect((1, 1, 0.85))
    title = ax.set_title("flat — the 2D picture you already know", color=INK, fontsize=9)
    # 3D axes need the margins set by hand; the default leaves the z-label hanging off.
    fig.subplots_adjust(left=0.0, right=0.97, top=0.93, bottom=0.03)

    # Plane corners in x, y; only the height and the far edge move.
    plane_x, plane_y = np.meshgrid(np.linspace(-2.2, 2.2, 2), np.linspace(-2.2, 2.2, 2))
    holder: dict = {"plane": None, "texts": []}

    def draw(i: int, progress: float) -> None:
        rise = min(1.0, progress / 0.6)
        plane_t = max(0.0, min(1.0, (progress - 0.6) / 0.4))
        cam = min(1.0, progress / 0.72)

        scat._offsets3d = (X[:, 0], X[:, 1], rise * height)
        hi._offsets3d = (_CALLOUTS[:, 0], _CALLOUTS[:, 1], rise * call_h)

        ax.view_init(elev=88 - 66 * cam, azim=-90 + 26 * progress)
        ax.set_zlabel("x3 = x1² + x2²" if cam > 0.35 else "")

        for t in holder["texts"]:
            t.remove()
        holder["texts"] = []
        if rise > 0.25:
            for (px, py), pz, label in zip(_CALLOUTS, rise * call_h, _CALLOUT_LABELS):
                holder["texts"].append(
                    ax.text(px, py, pz + 0.30, label, color=AMBER, fontsize=7, zorder=10,
                            ha="center")
                )

        if holder["plane"] is not None:
            holder["plane"].remove()
            holder["plane"] = None
        if plane_t > 0:
            # The sheet slides in from the right: its left edge sweeps across to fill the gap.
            left = 2.2 - 4.4 * plane_t
            px = np.where(plane_x < left, left, plane_x)
            holder["plane"] = ax.plot_surface(
                px, plane_y, np.full_like(plane_x, _PLANE_Z),
                color=ACCENT, alpha=0.35, shade=False, zorder=1,
            )

        if rise < 0.02:
            title.set_text("flat — the 2D picture you already know")
        elif rise < 1.0:
            title.set_text("lifting: every point climbs to x3 = x1² + x2²")
        elif plane_t < 1.0:
            title.set_text("a flat plane slides through the gap")
        else:
            title.set_text("one flat cut — impossible in 2D, easy up here")

    data = anim.gif_bytes(fig, draw, frames=32, fps=fps, hold_first=4, hold_last=7)
    plt.close(fig)
    return data


def road_gif_bytes(fps: int = 12) -> bytes:
    """A separating line growing its margin outward until it jams against the closest dots.

    The road starts as a thin band hugging the blue class, then its far edge sweeps across
    the empty gap. It cannot grow forever: the moment each edge reaches the nearest point,
    the widening stops and those points light up as the support vectors. That stop is the
    whole idea of a widest-road classifier.
    """
    use_house_style()

    import matplotlib.pyplot as plt

    from kidsml.plots import scatter_2d
    from kidsml.trees import fit_linear_svm

    model, X, y = fit_linear_svm(remove="none")
    w = model.coef_[0]
    b = model.intercept_[0]
    support = model.support_vectors_

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    fig.patch.set_facecolor("black")
    scatter_2d(X, y, ax=ax)
    ax.set_xlim(-2.7, 2.9)
    ax.set_ylim(-2.3, 2.3)
    xs = np.linspace(-2.7, 2.9, 60)

    def edge_y(c: float) -> np.ndarray:
        # Points where w·(x, y) + b = c, solved for y.
        return (c - b - w[0] * xs) / w[1]

    blue_edge = edge_y(-1.0)
    (moving_line,) = ax.plot([], [], color=ACCENT, linewidth=2.0, zorder=4)
    (centre_line,) = ax.plot([], [], color=ACCENT, linewidth=1.2, linestyle=":", alpha=0.0, zorder=4)
    rings = ax.scatter(
        support[:, 0], support[:, 1], s=170,
        facecolors="none", edgecolors=AMBER, linewidths=2.2, alpha=0.0, zorder=5,
    )
    title = ax.set_title("the road hugs the blue class — no shoulder", color=INK, fontsize=9)
    holder: dict = {"band": None}

    def draw(i: int, progress: float) -> None:
        top_c = -1.0 + 2.0 * progress  # far edge sweeps from the blue line across to red
        top_edge = edge_y(top_c)
        centre_edge = edge_y(-1.0 + progress)

        moving_line.set_data(xs, top_edge)
        centre_line.set_data(xs, centre_edge)
        centre_line.set_alpha(0.9 * progress)

        if holder["band"] is not None:
            holder["band"].remove()
        holder["band"] = ax.fill_between(
            xs, blue_edge, top_edge, color=ACCENT, alpha=0.16, zorder=2
        )

        # The rings appear as the edges jam against the closest points.
        rings.set_alpha(max(0.0, min(1.0, (progress - 0.82) / 0.18)))

        if progress < 0.35:
            title.set_text("the road hugs the blue class — no shoulder")
        elif progress < 0.98:
            title.set_text("widening: the far edge sweeps across the empty gap")
        else:
            title.set_text("stop — both edges jam against the support vectors")

    data = anim.gif_bytes(fig, draw, frames=30, fps=fps, hold_first=4, hold_last=8)
    plt.close(fig)
    return data
