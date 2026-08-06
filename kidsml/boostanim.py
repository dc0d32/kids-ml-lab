"""Watch the boosting staircase build itself, one stair at a time.

The chapter's teaching point is that the fixes are *cumulative*: each tiny tree adds one more
wooden stair to the running total, and the total creeps toward the dots. A slider over the
number of rounds shows one frozen frame at a time and asks the reader to stitch the progression
together in their head. This clip does the stitching for them.

The green prediction line starts as one flat step and gains stairs round by round until it
traces the curve through the blue dots. Boosting does most of its moving in the first handful of
rounds and then only nudges, so the frames are spaced by equal *change in the line* — early
rounds each get their own frame, later near-identical rounds share one — using
:func:`kidsml.anim.travel_spaced`. Every frame then moves the picture by about the same amount.

The trace comes from :func:`kidsml.trees.boosting_trace`, the same function the page and
notebook already use, so the clip and the slider are drawing the identical run. Frames are
stitched under one shared colour palette, the way ``kidsml/foldspace.py`` does, so the green
line stays a clean green instead of ghosting frame to frame.
"""

from __future__ import annotations

import io

import numpy as np

from kidsml import anim
from kidsml.plots import ACCENT, COOL, INK, PANEL, use_house_style
from kidsml.trees import boosting_trace


def staircase_gif_bytes(
    learning_rate: float = 0.25,
    max_depth: int = 1,
    seed: int = 4,
    n_steps: int = 45,
    frames: int = 34,
    fps: int = 12,
    hold_first: int = 4,
    hold_last: int = 8,
) -> bytes:
    """Return an animated GIF of the boosting running total growing, as raw bytes.

    ``learning_rate``, ``max_depth`` and ``seed`` match the page's staircase slider so the clip
    and the slider show the same run. The line is drawn once per selected round; the rounds are
    chosen by equal distance travelled, so the busy early rounds each get a frame and the quiet
    late ones do not. Roughly thirty frames at a small figure keeps the build well inside the
    chapter's wall-clock budget.
    """
    use_house_style()

    import matplotlib.pyplot as plt
    from PIL import Image

    trace = boosting_trace(n_steps=n_steps, learning_rate=learning_rate, max_depth=max_depth, seed=seed)
    x_grid = trace["x_grid"]
    grids = np.array([stage["running_grid"] for stage in trace["stages"]])
    picked = anim.travel_spaced(grids, frames)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("black")

    ax.scatter(trace["x"], trace["y"], s=42, color=COOL, edgecolors=PANEL, linewidths=0.6, zorder=2)
    (line,) = ax.plot(x_grid, grids[picked[0]], color=ACCENT, linewidth=2.6, zorder=3)

    ax.set_xlim(x_grid.min(), x_grid.max())
    pad = 0.15 * (trace["y"].max() - trace["y"].min())
    ax.set_ylim(trace["y"].min() - pad, trace["y"].max() + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("prediction")
    title = ax.set_title("running total after 1 stair", color=INK)
    fig.tight_layout()

    def draw(i: int, _progress: float) -> None:
        # One frame per boosting round, so a stair is added at a steady visible rate.
        k = min(max(0, i - hold_first), len(picked) - 1)
        step = int(picked[k]) + 1
        line.set_ydata(grids[picked[k]])
        stairs = "1 stair" if step == 1 else f"{step} stairs"
        title.set_text(f"running total after {stairs}")

    data = anim.gif_bytes(
        fig, draw, frames=len(picked), fps=fps,
        hold_first=hold_first, hold_last=hold_last,
        first_pause_ms=700, last_pause_ms=1600,
    )
    plt.close(fig)
    return data
