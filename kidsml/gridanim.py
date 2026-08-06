"""Watch a matrix *move* space — the two claims Chapter 12 asks you to believe.

Chapter 12's whole thesis is that a matrix is not a box of numbers, it is an instruction
for moving space. Two of its claims are the kind you only believe once you have watched
them happen:

* **Two moves fuse into one.** Doing ``A`` and then ``B`` lands every point exactly where
  the single matrix ``B @ A`` would. A tiny "largest disagreement: 1e-15" number does not
  make a reader feel that. Two grids racing to the same finish does.
* **Collapse throws information away.** The ``collapse onto a line`` preset has determinant
  zero: two dimensions become one, and there is no way back. You feel the irreversibility
  when you watch a whole house flatten into a line segment.

Both clips morph the grid **honestly**. To go from the identity to a matrix ``M`` we
interpolate the matrix entries, ``(1 - t) * I + t * M``, and apply that to the original
grid points. Every in-between frame is therefore a genuine linear map of the starting
grid — a real matrix, not a shape tween.

The little house from :func:`kidsml.linalg.house` rides along in both clips, because a
house makes a rotation, a flip or a shear obvious in a way a bare grid never could. It is
drawn in the brightest colour on the page (``SHAPE``); the un-moved starting grid sits
underneath as the faint ``GHOST`` layer.

Frames are stitched into a looping GIF entirely in memory by :func:`kidsml.anim.gif_bytes`.
Use the bytes with ``st.image(...)`` on a page or ``IPython.display.Image(data=...)`` in a
notebook.
"""

from __future__ import annotations

import numpy as np

from kidsml import anim
from kidsml import linalg as la
from kidsml.plots import COOL, EDGE, GHOST, SHAPE, use_house_style

_I = np.eye(2)


# The two matrices the "two moves become one" clip races. They match the pair the page and
# notebook already hand to `chained_equals_single`, so the animation shows the same claim
# the sliders and the table below it let the reader poke at.
CHAIN_A = la.matrix(1.0, 0.6, 0.0, 1.0)
CHAIN_B = la.matrix(0.7, -0.8, 0.8, 0.7)


def _interp(M: np.ndarray, t: float) -> np.ndarray:
    """A genuine matrix part-way from the identity to ``M``: ``(1 - t) * I + t * M``."""
    return (1.0 - t) * _I + t * np.asarray(M, dtype=float)


def _apply_chain(points: np.ndarray, mats) -> np.ndarray:
    """Send ``points`` through a list of matrices, in order (first in the list first)."""
    out = points
    for M in mats:
        out = la.apply_matrix(out, M)
    return out


def _bound(house: np.ndarray, mats_list, floor: float = 2.6, cap: float = 4.2) -> float:
    """A symmetric axis limit big enough to hold the house in every state it visits."""
    reach = float(np.max(np.abs(house)))
    for mats in mats_list:
        reach = max(reach, float(np.max(np.abs(_apply_chain(house, mats)))))
    return float(min(cap, max(floor, reach * 1.15)))


def _scene(limit: float, step: float):
    """Build the shared black stage: faint ghost grid, blank moving grid, blank house.

    Returns the figure plus the artists that later frames move. The straight grid lines
    only need their two endpoints — a linear map keeps them straight — so the grid is cheap
    to redraw even across fifty-odd frames.
    """
    use_house_style()

    import matplotlib.pyplot as plt

    base = la.grid_lines(limit=limit, step=step, points_per_line=2)
    house = la.house()

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color=EDGE, linewidth=1.0, antialiased=False, zorder=0)
    ax.axvline(0, color=EDGE, linewidth=1.0, antialiased=False, zorder=0)

    for line in base:
        ax.plot(line[:, 0], line[:, 1], color=GHOST, linewidth=0.8, antialiased=False, zorder=1)

    moving = [ax.plot([], [], color=COOL, linewidth=1.3, antialiased=False, zorder=2)[0] for _ in base]
    (house_line,) = ax.plot([], [], color=SHAPE, linewidth=2.6, antialiased=False, zorder=3)
    title = ax.set_title("", color=SHAPE, fontsize=11)
    # The title is empty when the layout is computed and changes on every frame, so
    # tight_layout reserves no room for it and the text is cut in half the moment it
    # appears. Give it the space up front instead.
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.05)
    return fig, ax, base, moving, house, house_line, title


def _paint(base, moving, house, house_line, mats) -> None:
    """Move every grid line and the house into the state given by ``mats``."""
    for line_pts, artist in zip(base, moving):
        moved = _apply_chain(line_pts, mats)
        artist.set_data(moved[:, 0], moved[:, 1])
    moved_house = _apply_chain(house, mats)
    house_line.set_data(moved_house[:, 0], moved_house[:, 1])


# ---------------------------------------------------------------------------
# Job 1 — two grid moves become one
# ---------------------------------------------------------------------------

# The moving section, phase by phase (frame counts). Journey one morphs under A, pauses,
# morphs under B, pauses on its finish; then journey two snaps back to the start and morphs
# once under the single matrix B @ A.
_A_MORPH = 10
_A_PAUSE = 5
_B_MORPH = 10
_BA_PAUSE = 6
_SNAP = 4
_C_MORPH = 10
_CHAIN_MOVING = _A_MORPH + _A_PAUSE + _B_MORPH + _BA_PAUSE + _SNAP + _C_MORPH


def _chain_state(k: int, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """The ``(mats, title)`` for frame ``k`` of the moving section (0-based)."""
    if k < _A_MORPH:
        t = anim.ease(k / (_A_MORPH - 1))
        return [_interp(A, t)], "Journey 1  ①  move by A"

    k -= _A_MORPH
    if k < _A_PAUSE:
        return [A], "Journey 1  ·  arrived at A — pause"

    k -= _A_PAUSE
    if k < _B_MORPH:
        t = anim.ease(k / (_B_MORPH - 1))
        return [A, _interp(B, t)], "Journey 1  ②  then move by B"

    k -= _B_MORPH
    if k < _BA_PAUSE:
        return [A, B], "Journey 1  ·  finished (A then B)"

    k -= _BA_PAUSE
    if k < _SNAP:
        return [_I], "Journey 2  ·  snap back to the start"

    k -= _SNAP
    t = anim.ease(k / (_C_MORPH - 1))
    return [_interp(C, t)], "Journey 2  ·  one move by  B @ A"


def chain_gif_bytes(A=CHAIN_A, B=CHAIN_B, step: float = 0.5, fps: int = 12) -> bytes:
    """Race two journeys and land them on the same grid.

    Journey one bends the grid under ``A``, holds so you can see where it reached, then
    bends again under ``B``. Journey two snaps the grid back to the start and bends it once
    under the single matrix ``B @ A``. Both finish on an identical grid — that identity is
    the whole claim, and it is a thing you watch rather than read off a table.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = B @ A

    house = la.house()
    limit = _bound(house, [[A], [A, B], [C]])
    fig, ax, base, moving, house, house_line, title = _scene(limit, step)

    hold_first = 4
    hold_last = 6

    def draw(i: int, _progress: float) -> None:
        if i < hold_first:
            mats, label = [_I], "Two journeys — same finish?"
        elif i >= hold_first + _CHAIN_MOVING:
            mats, label = [C], "Same place!   A then B  =  B @ A"
        else:
            mats, label = _chain_state(i - hold_first, A, B, C)
        _paint(base, moving, house, house_line, mats)
        title.set_text(label)

    data = anim.gif_bytes(
        fig,
        draw,
        frames=_CHAIN_MOVING,
        fps=fps,
        hold_first=hold_first,
        hold_last=hold_last,
    )

    import matplotlib.pyplot as plt

    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# Job 2 — collapse throws information away
# ---------------------------------------------------------------------------


def collapse_gif_bytes(preset: str = "collapse onto a line", step: float = 0.5, fps: int = 12) -> bytes:
    """Flatten the whole grid and the house down onto a single line, and hold there.

    The preset's determinant is zero, so two dimensions become one. The house — a shape
    with a clear roof and a clear door — becomes a line segment, and nothing in the picture
    can tell you which way was up any more. That is what "no way back" looks like.
    """
    C = la.matrix(*la.PRESETS[preset])

    house = la.house()
    limit = _bound(house, [[C]], floor=2.2)
    fig, ax, base, moving, house, house_line, title = _scene(limit, step)

    def draw(_i: int, progress: float) -> None:
        _paint(base, moving, house, house_line, [_interp(C, progress)])
        if progress <= 0.0:
            title.set_text("A full house, a full grid")
        elif progress >= 1.0:
            title.set_text("Flat. Two dimensions became one — no way back")
        else:
            title.set_text("squashing everything onto one line…")

    data = anim.gif_bytes(
        fig,
        draw,
        frames=22,
        fps=fps,
        hold_first=4,
        hold_last=8,
    )

    import matplotlib.pyplot as plt

    plt.close(fig)
    return data
