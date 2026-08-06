"""Watch the sliding window build a feature map, one landing at a time.

Chapter 19 tells the reader that a convolution is one little window that *slides* across a
picture — "like a tiny inspector" — visiting nine legal landing pads and dropping one
number at each. The old page then showed three static panels and the window never moved,
so the single most important verb in the chapter happened entirely in the reader's head.

So we animate it. A 3×3 window hops across the 5×5 image, left to right and then down a
row. On every landing we show the nine products sitting under the window (each image
number times the kernel number on top of it), and the moment those nine add up, the
matching cell of the 3×3 output grid lights up with their sum. By the last frame all nine
output cells are full and the reader has watched the feature map get *built* instead of
being handed the finished thing.

The numbers are exactly the by-hand example from the chapter: the 5×5 vertical-edge image
and the vertical-edge kernel, so the motion matches the arithmetic the reader just did.
"""

from __future__ import annotations

import numpy as np

from kidsml import anim
from kidsml.datasets import tiny_image
from kidsml.plots import (
    ACCENT,
    BACKGROUND,
    COUNT_CMAP,
    EDGE,
    INK,
    PANEL,
    PIXEL_CMAP,
    REGION_CMAP,
    use_house_style,
)

# The kernel is the chapter's fixed vertical-edge grid. It is spelled out here rather than
# imported from ``vision`` so this small clip stays free of the heavy image-model imports.
_KERNEL = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)

# Row-major order of the nine landing pads: across a row, then down to the next row.
_PADS = [(r, c) for r in range(3) for c in range(3)]


def _output(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = np.zeros((3, 3), dtype=float)
    for r, c in _PADS:
        out[r, c] = float((image[r:r + 3, c:c + 3] * kernel).sum())
    return out


def slide_gif_bytes(frames_per_pad: int = 4, fps: int = 10) -> bytes:
    """Return an animated GIF of the window sliding across the image, as raw bytes.

    ``frames_per_pad`` is how many frames each of the nine landings holds still for, so the
    reader has time to read the products and the sum before the window hops on. Nine pads
    at four frames each, plus a beat held at each end, is roughly 45 small frames — well
    inside the chapter's wall-clock budget on a laptop CPU.
    """
    use_house_style()

    import matplotlib.pyplot as plt

    image = tiny_image()
    kernel = _KERNEL
    output = _output(image, kernel)

    fig = plt.figure(figsize=(9.4, 3.6))
    fig.patch.set_facecolor(BACKGROUND)
    # Three fixed square boxes. Squares of equal size, hand-placed, so equal-aspect images
    # never fight an automatic layout — the boxes already match the data, so nothing shifts
    # from frame to frame.
    box_h = 0.70
    box_w = box_h * 3.6 / 9.4
    ax_img = fig.add_axes([0.045, 0.15, box_w, box_h])
    ax_prod = fig.add_axes([0.395, 0.15, box_w, box_h])
    ax_out = fig.add_axes([0.72, 0.15, box_w, box_h])

    # --- Left panel: the 5×5 image, drawn once, with a window that moves. ---
    ax_img.imshow(image, cmap=PIXEL_CMAP, interpolation="nearest", vmin=0, vmax=9)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.grid(False)
    ax_img.set_title("image + window", color=INK)
    img_hi = image.max()
    for (r, c), v in np.ndenumerate(image):
        ax_img.text(
            c, r, f"{v + 0.0:.0f}".replace("-0", "0"), ha="center", va="center", fontsize=9,
            color=BACKGROUND if v > 0.6 * img_hi else INK,
        )
    from matplotlib.patches import Rectangle

    window = Rectangle(
        (-0.5, -0.5), 3, 3, fill=False, edgecolor=ACCENT, linewidth=3.0, zorder=5
    )
    ax_img.add_patch(window)

    kmax = float(np.abs(kernel).max()) or 1.0

    def _dress(ax) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)
        ax.set_aspect("equal")

    def draw(i: int, progress: float) -> None:
        moving = 9 * frames_per_pad
        moving_i = min(max(i - anim_hold_first, 0), moving - 1)
        pad = min(8, moving_i // frames_per_pad)
        r, c = _PADS[pad]
        patch = image[r:r + 3, c:c + 3]
        products = patch * kernel
        value = output[r, c]

        window.set_xy((c - 0.5, r - 0.5))

        # --- Middle panel: the nine products under the window right now. ---
        ax_prod.clear()
        _dress(ax_prod)
        # Background shows the kernel's sign: the +1 column red, the -1 column blue, so the
        # reader sees which column the bright pixels are about to hit.
        ax_prod.imshow(kernel, cmap=REGION_CMAP, vmin=-kmax, vmax=kmax, interpolation="nearest")
        for (pr, pc), pv in np.ndenumerate(products):
            ax_prod.text(pc, pr, f"{pv:.0f}".replace("-0", "0"), ha="center", va="center",
                         fontsize=11, color=INK)
        ax_prod.set_title("nine products", color=INK)
        ax_prod.set_xlabel(f"they add up to {value:.0f}".replace("-0", "0"), color=INK, fontsize=11)

        # --- Right panel: the output grid, filling in as the window lands. ---
        ax_out.clear()
        _dress(ax_out)
        shown = np.full((3, 3), np.nan)
        for done in range(pad + 1):
            dr, dc = _PADS[done]
            shown[dr, dc] = output[dr, dc]
        masked = np.ma.masked_invalid(shown)
        cmap = COUNT_CMAP.with_extremes(bad=PANEL)
        ax_out.imshow(
            masked, cmap=cmap, vmin=float(output.min()), vmax=float(output.max()),
            interpolation="nearest",
        )
        for (orow, ocol), ov in np.ndenumerate(shown):
            if not np.isnan(ov):
                ax_out.text(ocol, orow, f"{ov:.0f}".replace("-0", "0"), ha="center", va="center",
                            fontsize=11, color=INK)
        # Outline all nine cells faintly, so the reader can see how many are still to
        # come. Without it the first frame is one lit square floating in the dark and
        # "nine landing pads" has nothing to land on.
        for orow in range(3):
            for ocol in range(3):
                ax_out.add_patch(Rectangle((ocol - 0.5, orow - 0.5), 1, 1, fill=False,
                                           edgecolor=EDGE, linewidth=1.0, zorder=4))

        # The cell that just lit up gets a green border, matching the window.
        landed = Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=ACCENT,
                           linewidth=2.5, zorder=5)
        ax_out.add_patch(landed)
        ax_out.set_title(f"output — landing {pad + 1} of 9", color=INK)

    anim_hold_first = 3
    data = anim.gif_bytes(
        fig, draw, frames=9 * frames_per_pad, fps=fps,
        hold_first=anim_hold_first, hold_last=6,
    )
    plt.close(fig)
    return data


def slide_still():
    """The finished result frozen: image + window at the last pad, and the full output.

    For a reader who wants to stop the motion and stare at where every number came from.
    """
    use_house_style()

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = tiny_image()
    kernel = _KERNEL
    output = _output(image, kernel)

    fig, (ax_img, ax_ker, ax_out) = plt.subplots(1, 3, figsize=(10.0, 3.4))
    fig.patch.set_facecolor(BACKGROUND)

    for ax, grid, title, cmap, vlim in (
        (ax_img, image, "image", PIXEL_CMAP, (0, 9)),
        (ax_ker, kernel, "kernel", REGION_CMAP, (-1, 1)),
        (ax_out, output, "3 by 3 output", COUNT_CMAP, (output.min(), output.max())),
    ):
        ax.imshow(grid, cmap=cmap, vmin=vlim[0], vmax=vlim[1], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, color=INK)
        hi = grid.max()
        lo = grid.min()
        for (r, c), v in np.ndenumerate(grid):
            shade = (v - lo) / (hi - lo + 1e-9)
            ax.text(c, r, f"{v:.0f}".replace("-0", "0"), ha="center", va="center", fontsize=10,
                    color=BACKGROUND if shade > 0.7 else INK)

    ax_img.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor=EDGE,
                               linewidth=1.5, zorder=5))
    fig.tight_layout()
    return fig
