"""Two short clips for the first two chapters, where the motion is the whole point.

Chapters 1 and 2 both hand the reader a finished answer and ask them to imagine the search
that found it. Chapter 1 prints the entire downhill path as a constellation of dots on a
static loss map; Chapter 2 shows a before/after pair of a line that has already swung round.
In both, the thing worth understanding — *the model does not know where the answer is, it
finds it by measuring and moving* — only reads when you watch it happen.

So we animate them, the same way Chapter 15's fold does it: draw matplotlib frames and
stitch them into a looping GIF in memory with :mod:`kidsml.anim`.

* :func:`descent_gif_bytes` — the two allowance-line numbers walking downhill into the
  valley, dots landing one at a time, with the current ``w``, ``b`` and error in the title
  so the moving dot is tied to the two numbers changing.
* :func:`correction_gif_bytes` — one perceptron update: the missed dog flashes, then the
  boundary rotates and slides until that dog sits on the correct side. The numbers are the
  exact ones the chapter works out by hand (``w = (1, 1)``, ``b = -20`` on the dog at
  ``(6, 5)``), so the motion matches the arithmetic.
"""

from __future__ import annotations

import numpy as np

from kidsml import anim
from kidsml.datasets import allowance, two_blobs_tiny
from kidsml.linear import gradient_descent_line
from kidsml.nn_numpy import perceptron_step
from kidsml.plots import ACCENT, COOL, COUNT_CMAP, GRIDLINE, INK, PANEL, SHAPE, WARM, loss_surface, use_house_style


def descent_gif_bytes(
    steps: int = 26,
    lr: float = 0.01,
    fps: int = 12,
) -> bytes:
    """Animate gradient descent walking down the allowance loss map.

    The dots land one at a time, tracing the path from the top of the hill at ``w = 0``,
    ``b = 0`` down into the valley. The title carries the current ``w``, ``b`` and error so
    the reader can tie the moving dot to the two numbers changing.

    One real gradient-descent step lands per frame, so the reader sees the honest shape of
    the walk: a big leap first, then shorter and shorter nudges as the dots crowd into the
    valley — which is exactly what "the steps get smaller near the bottom" looks like.
    """
    use_house_style()

    import matplotlib.pyplot as plt

    weeks, dollars = allowance()
    path = gradient_descent_line(weeks, dollars, w=0.0, b=0.0, lr=lr, steps=steps)
    sw = path["w"]
    sb = path["b"]
    sloss = path["loss"]

    W, B, Z = loss_surface(weeks, dollars, w_range=(0, 6), b_range=(-5, 15))

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    fig.patch.set_facecolor("black")
    ax.contourf(W, B, Z, levels=24, cmap=COUNT_CMAP)
    ax.contour(W, B, Z, levels=12, colors=[GRIDLINE], linewidths=0.7)

    trail = ax.scatter([], [], s=16, color=SHAPE, edgecolors=PANEL, linewidths=0.4, zorder=4)
    head = ax.scatter([], [], s=110, color=ACCENT, edgecolors=PANEL, linewidths=1.2, zorder=5)

    ax.set_xlim(0, 6)
    ax.set_ylim(-5, 15)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    title = ax.set_title("", color=INK)
    # The title changes every frame, so it has to be given room once, up front: a
    # tight_layout computed against an empty title clips the text the moment it appears.
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.12)

    n = len(sw)
    hold_first = 4
    frames = n

    def draw(i: int, progress: float) -> None:
        # Reveal one gradient-descent step per frame so the walk lands steadily, one dot at
        # a time, rather than in the bunched-up rush an eased progress would give.
        if i < hold_first:
            idx = 0
        elif i < hold_first + frames:
            idx = min(i - hold_first, n - 1)
        else:
            idx = n - 1
        trail.set_offsets(np.column_stack([sw[: idx + 1], sb[: idx + 1]]))
        head.set_offsets([[sw[idx], sb[idx]]])
        title.set_text(f"w = {sw[idx]:.2f}    b = {sb[idx]:.2f}    error = {sloss[idx]:.0f}")

    data = anim.gif_bytes(fig, draw, frames=frames, fps=fps, hold_first=hold_first, hold_last=8)
    plt.close(fig)
    return data


def correction_gif_bytes(frames: int = 28, fps: int = 12) -> bytes:
    """Animate one perceptron correction on the ten-dogs data.

    The missed red dog at ``(6, 5)`` flashes, then the boundary rotates and shifts from the
    too-strict line ``w = (1, 1)``, ``b = -20`` to its corrected place ``w = (7, 6)``,
    ``b = -19`` — the exact update the chapter does by hand. The dog ends up on the red side.

    Only one correction is shown on purpose: the lesson here is what a *single* update does,
    not the whole training run.
    """
    use_house_style()

    import matplotlib.pyplot as plt

    X, y = two_blobs_tiny()
    w_bad = np.array([1.0, 1.0])
    b_bad = -20.0
    missed = X[5]
    w_after, b_after, _ = perceptron_step(w_bad, b_bad, X[5], y[5])

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    fig.patch.set_facecolor("black")

    colours = np.where(y == 1, WARM, COOL)
    ax.scatter(X[:, 0], X[:, 1], s=90, c=colours, edgecolors=PANEL, linewidths=0.8, zorder=3)
    circle = ax.scatter(
        [missed[0]], [missed[1]], s=260, facecolors="none", edgecolors=ACCENT,
        linewidths=2.6, zorder=4,
    )
    (line,) = ax.plot([], [], color=ACCENT, linewidth=2.4, zorder=2)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.set_xlabel("how tall (hand-spans)")
    ax.set_ylabel("how heavy (bags of sugar)")
    title = ax.set_title("", color=INK)
    # The title changes every frame, so it has to be given room once, up front: a
    # tight_layout computed against an empty title clips the text the moment it appears.
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.12)

    xs = np.array([0.0, 12.0])
    hold_first = 6

    def endpoints(w1: float, w2: float, b: float) -> np.ndarray:
        return -(w1 * xs + b) / w2

    # Interpolate the line's on-screen endpoints, not its raw weights. Blending the weights
    # linearly makes the boundary lurch — almost all of the visible travel happens in the
    # first few frames, because the score field it comes from is not linear in the weights.
    # Sliding the two endpoints instead sweeps the line across the frame at an even pace.
    y_before = endpoints(w_bad[0], w_bad[1], b_bad)
    y_after = endpoints(w_after[0], w_after[1], b_after)

    def draw(i: int, progress: float) -> None:
        line.set_data(xs, (1.0 - progress) * y_before + progress * y_after)

        if i < hold_first:
            # Flash the missed dog to say "this one is on the wrong side".
            on = i % 2 == 0
            circle.set_alpha(1.0 if on else 0.25)
            circle.set_sizes([340 if on else 200])
            title.set_text("this red dog is on the wrong side")
        elif progress < 1.0:
            circle.set_alpha(1.0)
            circle.set_sizes([260])
            title.set_text("add the dog to the weights — the line swings")
        else:
            circle.set_alpha(1.0)
            circle.set_sizes([260])
            title.set_text("now the red dog is on the red side")

    data = anim.gif_bytes(fig, draw, frames=frames, fps=fps, hold_first=hold_first, hold_last=8)
    plt.close(fig)
    return data
