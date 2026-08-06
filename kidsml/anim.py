"""Short animations, made the same way everywhere.

Some ideas in this course are about *change over time*, and a before/after pair asks the
reader to do the animating in their head. A grid bending under a matrix, a point walking
downhill, a window sliding across a picture, cluster centres drifting into place — in every
one of those the motion **is** the teaching point.

The recipe, which Chapter 15's fold established and this module now shares:

* Draw the frames with matplotlib and stitch them into a GIF with Pillow, entirely in
  memory. Both are already dependencies; nothing new is installed and nothing hits disk.
* Keep clips short — roughly 25 to 45 frames. They have to build inside the chapter's
  wall-clock budget (``tests/test_notebooks.py``) on a laptop CPU, and a long clip is a
  clip the reader stops watching.
* Hold the first and last frames. A looping GIF with no pauses reads as a flicker; a beat
  at each end lets the reader see what changed.
* Ease the motion. Linear movement looks mechanical, and the eye loses track of a point
  that starts at full speed.

A GIF loops forever on its own, which is what we want: the reader watches it three times
without having to press anything.

Usage::

    def draw(i, alpha):
        scatter.set_offsets(start + alpha * (end - start))
        title.set_text("folding..." if alpha < 1 else "done")

    data = anim.gif_bytes(fig, draw, frames=30)

Then ``st.image(data)`` on a page, or ``IPython.display.Image(data=data)`` in a notebook.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np

FrameDrawer = Callable[[int, float], None]


def ease(t: float) -> float:
    """Smoothstep. Start slow, speed up, ease out — motion the eye can follow."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def gif_bytes(
    fig,
    draw: FrameDrawer,
    frames: int = 30,
    fps: int = 12,
    hold_first: int = 4,
    hold_last: int = 6,
    first_pause_ms: int = 900,
    last_pause_ms: int = 1600,
) -> bytes:
    """Render an animation to a looping GIF, in memory.

    ``draw(index, progress)`` is called once per frame and should move the artists that
    were already added to ``fig``. ``progress`` runs 0 → 1 across the moving section, eased,
    and stays pinned at 0 during the opening hold and at 1 during the closing hold.

    The held frames are real frames, not a trick: some clips want to change what the title
    says while the picture is still, and a few of them fade something in at the end.
    """
    from PIL import Image

    total = hold_first + frames + hold_last
    images = []

    for i in range(total):
        if i < hold_first:
            progress = 0.0
        elif i < hold_first + frames:
            progress = ease((i - hold_first) / max(1, frames - 1))
        else:
            progress = 1.0

        draw(i, progress)
        fig.canvas.draw()
        images.append(
            Image.frombuffer(
                "RGBA",
                fig.canvas.get_width_height(),
                fig.canvas.buffer_rgba(),
                "raw",
                "RGBA",
                0,
                1,
            ).convert("P", palette=Image.ADAPTIVE, colors=128)
        )

    durations = [1000 / fps] * total
    if hold_first:
        durations[hold_first - 1] = first_pause_ms
    durations[-1] = last_pause_ms

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    return buffer.getvalue()


def travel_spaced(path, count: int):
    """Pick ``count`` points from ``path`` spaced by equal *distance travelled*.

    Training paths are almost never evenly paced: a model sits on a plateau for hundreds of
    steps and then moves a long way in ten. Sampling every Nth step puts nearly all of the
    visible change into the first second of the clip and then shows a still picture. This
    samples along the path instead of along the clock, so every frame moves the picture by
    about the same amount.

    ``path`` is an ``(n_steps, n_values)`` array. Returns the chosen indices.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim == 1:
        path = path.reshape(-1, 1)

    hops = np.linalg.norm(np.diff(path, axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(hops)])
    if travelled[-1] <= 0:
        return np.linspace(0, len(path) - 1, count).astype(int)

    wanted = np.linspace(0.0, travelled[-1], count)
    return np.clip(np.searchsorted(travelled, wanted), 0, len(path) - 1)
