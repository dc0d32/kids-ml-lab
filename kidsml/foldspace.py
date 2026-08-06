"""Watch the hidden layer fold XOR into a space where one line works.

Chapter 15 tells the reader that a hidden layer does not bend the output line — it moves
the points into a new space where a straight line already works. That is easy to assert
and hard to believe from two static scatter plots side by side.

So we animate it. The four XOR corners start in the original ``(x1, x2)`` square and slide
to their hand-made hidden coordinates ``(OR-ish, AND-ish)``. The two red corners, which
began in opposite corners, travel to almost the same spot; the two blue corners stay put,
far apart. At the end one straight line drops in and separates them.

The numbers here are exactly the by-hand table in the chapter, so the motion matches the
arithmetic the reader just did.
"""

from __future__ import annotations

import io

import numpy as np

from kidsml.datasets import xor_exact
from kidsml.nn_numpy import MLP, mse
from kidsml.plots import ACCENT, COOL, INK, PANEL, WARM, use_house_style

# The four XOR corners, and where the two hand-made hidden features send them.
# OR-ish = h1, AND-ish = h2, straight from the chapter's table.
_ORIG = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
# Both red corners land on (1, 0). We split them by a hair so both stay visible instead
# of stacking into one dot — they are still, to the pencil, the same point.
_HIDDEN = np.array([[0.0, 0.0], [1.0, 0.06], [1.0, -0.06], [1.0, 1.0]])
_XOR = np.array([0, 1, 1, 0])


def _ease(t: float) -> float:
    """Smoothstep: start slow, speed up, ease out. Motion the eye can follow."""
    return t * t * (3.0 - 2.0 * t)


def _positions(alpha: float) -> np.ndarray:
    return (1.0 - alpha) * _ORIG + alpha * _HIDDEN


def fold_gif_bytes(hold: int = 5, morph: int = 22, settle: int = 8, fps: int = 12) -> bytes:
    """Return an animated GIF of the XOR fold as raw bytes.

    The clip is small and short on purpose: a few held frames at the start, a smooth
    morph, then a few frames where the separating line fades in. Roughly 35 frames at a
    small figure size keeps it well inside the notebook's wall-clock budget.

    Frames are drawn straight into memory and stitched into a GIF with Pillow, so nothing
    ever touches the disk.
    """
    use_house_style()

    import matplotlib.pyplot as plt
    from PIL import Image

    colours = np.where(_XOR == 1, WARM, COOL)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    fig.patch.set_facecolor("black")

    scat = ax.scatter(
        _ORIG[:, 0], _ORIG[:, 1], c=colours, s=170, edgecolor=PANEL, linewidth=1.5, zorder=3
    )
    (line,) = ax.plot([], [], color=ACCENT, linewidth=2.4, zorder=2)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.45, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("x1  →  h1 (OR-ish)")
    ax.set_ylabel("x2  →  h2 (AND-ish)")
    title = ax.set_title("original space", color=INK)
    fig.tight_layout()

    # The separating line score = h1 - 2*h2 - 0.5 = 0, i.e. h2 = (h1 - 0.5) / 2.
    hs = np.array([-0.35, 1.35])
    line_ys = (hs - 0.5) / 2.0

    total = hold + morph + settle
    frames = []
    for i in range(total):
        if i < hold:
            alpha = 0.0
            title.set_text("original space — red in opposite corners")
        elif i < hold + morph:
            alpha = _ease((i - hold) / (morph - 1))
            title.set_text("folding into the hidden space…")
        else:
            alpha = 1.0
            title.set_text("hidden space — one line splits red from blue")
        scat.set_offsets(_positions(alpha))

        if i >= hold + morph:
            k = (i - hold - morph + 1) / settle
            line.set_data(hs, line_ys)
            line.set_alpha(min(1.0, k))
        else:
            line.set_alpha(0.0)

        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                        fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("P"))

    plt.close(fig)

    buffer = io.BytesIO()
    durations = [1000 / fps] * total
    durations[hold - 1] = 900          # linger on the "before"
    durations[-1] = 1600               # and hold the "after" so the line reads
    frames[0].save(
        buffer, format="GIF", save_all=True, append_images=frames[1:],
        duration=durations, loop=0, disposal=2,
    )
    return buffer.getvalue()


def fold_still_pair():
    """A 'before / after' still pair, for readers who miss the motion in the GIF."""
    use_house_style()

    import matplotlib.pyplot as plt

    colours = np.where(_XOR == 1, WARM, COOL)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    panels = ((0.0, "before: original x1, x2 space"), (1.0, "after: hidden h1, h2 space"))
    for ax, (alpha, title) in zip(axes, panels):
        pts = _positions(alpha)
        ax.scatter(pts[:, 0], pts[:, 1], c=colours, s=150, edgecolor=PANEL, linewidth=1.4, zorder=3)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-0.45, 1.35)
        ax.set_aspect("equal")
        ax.set_title(title)
        if alpha >= 0.999:
            hs = np.array([-0.35, 1.35])
            ax.plot(hs, (hs - 0.5) / 2.0, color=ACCENT, linewidth=2.2, zorder=2)
            ax.set_xlabel("h1 = OR-ish")
            ax.set_ylabel("h2 = AND-ish")
        else:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
    fig.tight_layout()
    return fig


def _flatten(model: MLP) -> np.ndarray:
    return np.concatenate([W.ravel() for W in model.Ws] + [b.ravel() for b in model.bs])


def learned_xor_snapshots(lr: float = 0.25, n: int = 40, epochs: int = 4000, seed: int = 2):
    """Train [2, 3, 1] on XOR and return snapshots spaced by equal *visible* change.

    XOR training sits on a long flat plateau and then breaks all at once, so snapshots
    taken every N steps pile almost all of the movement into the far left of a scrub
    slider. Instead we train densely, measure how far the weights actually travelled
    between steps, and pick ``n`` snapshots that are equally spaced along that path. Now
    every drag of the slider moves the picture by about the same amount.
    """
    X, y = xor_exact()
    y_col = np.asarray(y).reshape(-1, 1)
    model = MLP([2, 3, 1], activation="tanh", seed=seed)

    snaps = []
    vectors = []
    for step in range(epochs + 1):
        snaps.append({
            "step": step,
            "Ws": [W.copy() for W in model.Ws],
            "bs": [b.copy() for b in model.bs],
            "loss": mse(model.forward(X), y_col),
        })
        vectors.append(_flatten(model))
        if step < epochs:
            model.step(X, y, lr=lr)

    vectors = np.asarray(vectors)
    hops = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(hops)])

    if travelled[-1] <= 0:
        return X, y, [snaps[0], snaps[-1]]

    targets = np.linspace(0.0, travelled[-1], n)
    picked = []
    seen = set()
    for t in targets:
        i = int(np.searchsorted(travelled, t))
        i = min(i, len(snaps) - 1)
        if i not in seen:
            seen.add(i)
            picked.append(snaps[i])
    if picked[-1]["step"] != epochs:
        picked.append(snaps[-1])
    return X, y, picked
