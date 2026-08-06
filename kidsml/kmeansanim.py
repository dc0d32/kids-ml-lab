"""Watch k-means actually converge, instead of clicking a step button a dozen times.

The algorithm is *named* after centres moving to the mean, but a step button only shows the
reader still snapshots. They have to imagine the drift between clicks. So we animate a whole
run: points recolour as each joins its nearest centre, then the centre markers glide to the
middle of their members, then it repeats, until nothing moves.

The two half-steps are the entire idea, so the title says which one is happening — "everyone
joins the nearest centre" while the dots recolour, "each centre moves to the middle of its
group" while the X markers travel. The centre positions are interpolated between rounds, so a
centre visibly walks from where it was to where its members pulled it rather than teleporting.

The per-round stages come straight from :func:`kidsml.unsupervised.kmeans_history`, so the
clip and the step button tell the exact same story.

The frames are stitched into a GIF under one shared colour palette. Chapter 15's fold clip
(``kidsml/foldspace.py``) stitches its own GIF for the same reason: a k-means frame carries
several saturated cluster colours at once, and giving each frame its own adaptive palette makes
those colours ghost from frame to frame. We reuse ``anim`` for the easing and the disk-free
build.
"""

from __future__ import annotations

import io

import numpy as np

from kidsml import anim
from kidsml.plots import (
    ACCENT,
    AMBER,
    COOL,
    GHOST,
    INK,
    PANEL,
    PINK,
    TEAL,
    VIOLET,
    WARM,
    use_house_style,
)
from kidsml.unsupervised import kmeans_history

# One colour per cluster, all from the house palette. k is small (2–5 on the page).
_CLUSTER_COLOURS = [COOL, WARM, ACCENT, AMBER, VIOLET, PINK, TEAL]

_ASSIGN_TITLE = "everyone joins the nearest centre"
_MOVE_TITLE = "each centre moves to the middle of its group"


def _colours_for(labels) -> list:
    if labels is None:
        return None
    return [_CLUSTER_COLOURS[i % len(_CLUSTER_COLOURS)] for i in labels]


def _build_plan(stages, k: int, assign_hold: int, move_frames: int):
    """A fully-specified frame list: (centres, point-colours, centre-colours, title).

    Stages alternate assign / move. An assign stage recolours the dots with the centres held
    still; a move stage keeps the colours and walks the centres to their new spot. We
    interpolate the move so the markers travel, and hold a few frames on each recolour so the
    join is readable before the drift starts.
    """
    plan = []
    start_centres = stages[0]["centres"]
    centre_colours = [_CLUSTER_COLOURS[i % len(_CLUSTER_COLOURS)] for i in range(k)]

    # Open on grey, unjoined dots so the first recolour is a change the eye can catch. The
    # centre markers already wear their cluster colours from the start.
    plan.append((start_centres, None, centre_colours, "k centres dropped in — no dot has joined yet"))

    for round_id in range(0, len(stages), 2):
        assign = stages[round_id]
        move = stages[round_id + 1] if round_id + 1 < len(stages) else None

        colours = _colours_for(assign["labels"])
        assign_centres = assign["centres"]
        for _ in range(assign_hold):
            plan.append((assign_centres, colours, centre_colours, _ASSIGN_TITLE))

        if move is None:
            continue

        target = move["centres"]
        if np.allclose(assign_centres, target, equal_nan=True):
            # Converged: the centres have nowhere left to walk. Say so and stop.
            for _ in range(assign_hold):
                plan.append((target, colours, centre_colours, "settled — nothing moves now"))
            break

        for f in range(move_frames):
            alpha = anim.ease((f + 1) / move_frames)
            centres = (1.0 - alpha) * assign_centres + alpha * target
            plan.append((centres, colours, centre_colours, _MOVE_TITLE))

    return plan


def kmeans_run_gif_bytes(
    k: int = 3,
    seed: int = 0,
    bad_start: bool = False,
    assign_hold: int = 4,
    move_frames: int = 9,
    fps: int = 12,
    hold_first: int = 4,
    hold_last: int = 8,
) -> bytes:
    """Return an animated GIF of a full k-means run as raw bytes.

    ``k``, ``seed`` and ``bad_start`` are passed straight to ``kmeans_history`` so the clip
    matches whatever the step button is showing. The clip recolours the dots, glides the
    centres, and repeats until the run converges — usually three or four rounds, which lands
    the whole thing comfortably inside the chapter's wall-clock budget.
    """
    use_house_style()

    import matplotlib.pyplot as plt
    from PIL import Image

    stages = kmeans_history(k=k, seed=seed, bad_start=bad_start)
    X = stages[0]["X"]
    plan = _build_plan(stages, k=len(stages[0]["centres"]), assign_hold=assign_hold, move_frames=move_frames)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    fig.patch.set_facecolor("black")

    points = ax.scatter(
        X[:, 0], X[:, 1], s=40, c=[GHOST] * len(X), edgecolors=PANEL, linewidths=0.6, zorder=2
    )
    centre_markers = ax.scatter(
        plan[0][0][:, 0],
        plan[0][0][:, 1],
        marker="X",
        s=340,
        c=plan[0][2],
        edgecolors=INK,
        linewidths=2.2,
        zorder=5,
    )

    ax.set_xlim(X[:, 0].min() - 0.6, X[:, 0].max() + 0.6)
    ax.set_ylim(X[:, 1].min() - 0.6, X[:, 1].max() + 0.6)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    title = ax.set_title(plan[0][3], color=INK)
    fig.tight_layout()

    def draw(i: int, _progress: float) -> None:
        # The plan already holds one entry per phase step (a recolour, or one slice of a
        # centre's glide), so the frame index picks a plan entry rather than easing.
        idx = min(max(0, i - hold_first), len(plan) - 1)
        frame_centres, point_colours, centre_colours, caption = plan[idx]
        centre_markers.set_offsets(frame_centres)
        centre_markers.set_facecolor(centre_colours)
        centre_markers.set_edgecolor(INK)
        points.set_color([GHOST] * len(X) if point_colours is None else point_colours)
        title.set_text(caption)

    data = anim.gif_bytes(
        fig, draw, frames=len(plan), fps=fps,
        hold_first=hold_first, hold_last=hold_last,
        first_pause_ms=900, last_pause_ms=1600,
    )
    plt.close(fig)
    return data
