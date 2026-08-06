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
    colors: int = 128,
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
        # `np.asarray(buffer_rgba())` rather than `Image.frombuffer(...,
        # canvas.get_width_height(), ...)`. On a HiDPI screen the macOS backend hands back
        # a buffer with twice as many pixels as `get_width_height()` reports, and reading
        # it at the reported size misaligns every row — the clip comes out ghosted and
        # sheared. The array knows its own shape, so this is right on every backend.
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()), "RGBA").convert("RGB"))

    # Every frame must be quantised against the *same* palette.
    #
    # Letting each frame pick its own adaptive palette looks fine one frame at a time and
    # produces a garish mess in the finished GIF: the file carries one global colour table,
    # so frame 12's indices get looked up in frame 0's colours. On a nearly-black clip the
    # palettes come out similar enough to hide it; on a colourful one — a loss surface, a
    # heat map — the picture turns to lurid red and green partway through.
    #
    # The shared palette is built from a strip of frames sampled across the whole clip, so
    # colours that only appear late still get a slot.
    sample_indices = sorted({int(round(t)) for t in np.linspace(0, total - 1, min(6, total))})
    strip = Image.new("RGB", (images[0].width, images[0].height * len(sample_indices)))
    for slot, index in enumerate(sample_indices):
        strip.paste(images[index], (0, images[0].height * slot))
    master = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

    images = [frame.quantize(palette=master, dither=Image.Dither.NONE) for frame in images]

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
        disposal=1,
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
