"""Linear algebra you can see.

The idea this module exists to support: **a matrix is not a box of numbers. It is an
instruction for moving space.** Every picture here shows the same grid before and after
some transformation, so "what a matrix does" stops being an abstraction.

The punchline the chapter builds to: stacking two linear moves gives you one linear move
(so extra layers buy nothing on their own), and putting a squish between them is what
finally bends the grid. That is the entire reason neural networks have activation
functions, and it is a fact you can see rather than take on trust.
"""

from __future__ import annotations

import numpy as np

from kidsml.nn_numpy import ACTIVATIONS

# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


def magnitude(v) -> float:
    """How long an arrow is. For [3, 4] this is 5 — the 3-4-5 triangle."""
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.sum(v * v)))


def direction(v) -> np.ndarray:
    """The same arrow squashed to length 1: which way it points, with the length removed."""
    v = np.asarray(v, dtype=float)
    length = magnitude(v)
    if length == 0:
        return v
    return v / length


def angle_degrees(v) -> float:
    """Which way the arrow points, measured anticlockwise from the x-axis."""
    v = np.asarray(v, dtype=float)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def dot(a, b) -> float:
    """How much two arrows agree.

    Positive means they broadly point the same way, zero means they are at right angles,
    negative means they disagree. This is the same arithmetic as a neuron's weighted sum,
    which is why a neuron can be read as "how much does this input look like the thing I
    am watching for?".
    """
    return float(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def project_onto(v, onto) -> np.ndarray:
    """The shadow of ``v`` cast along ``onto``.

    Shine a light straight down onto the line through ``onto``; this is where ``v`` lands.
    Chapter 21's PCA is this exact operation, with the line chosen to lose the least.
    """
    v = np.asarray(v, dtype=float)
    onto = np.asarray(onto, dtype=float)
    denominator = np.dot(onto, onto)
    if denominator == 0:
        return np.zeros_like(v)
    return onto * (np.dot(v, onto) / denominator)


# ---------------------------------------------------------------------------
# Matrices as movements of the grid
# ---------------------------------------------------------------------------


def matrix(a: float, b: float, c: float, d: float) -> np.ndarray:
    """Build ``[[a, b], [c, d]]``.

    Read it by columns, not rows: column one is where the arrow (1, 0) ends up, and
    column two is where (0, 1) ends up. Once you know where those two land, you know
    where *everything* lands — that is the whole content of "linear".
    """
    return np.array([[a, b], [c, d]], dtype=float)


PRESETS: dict[str, tuple[float, float, float, float]] = {
    "do nothing": (1, 0, 0, 1),
    "stretch sideways": (2, 0, 0, 1),
    "squash flat": (1, 0, 0, 0.3),
    "rotate a quarter turn": (0, -1, 1, 0),
    "shear (push the top over)": (1, 1, 0, 1),
    "mirror left-right": (-1, 0, 0, 1),
    "shrink everything": (0.5, 0, 0, 0.5),
    "collapse onto a line": (1, 2, 2, 4),
}

PRESET_BLURB = {
    "do nothing": "The grid stays exactly where it was. Every square is still a square.",
    "stretch sideways": "Twice as wide, same height. Squares become rectangles.",
    "squash flat": "The grid gets pressed down. Areas shrink.",
    "rotate a quarter turn": "The whole grid spins. Nothing is stretched — squares stay squares.",
    "shear (push the top over)": "Like sliding a deck of cards. Squares lean into parallelograms.",
    "mirror left-right": "The grid flips over. Watch the area number go negative.",
    "shrink everything": "Same shape, half the size. Area drops to a quarter.",
    "collapse onto a line": "Everything lands on one line. Two dimensions become one, and there is no way back.",
}


def where_the_arrows_land(M) -> dict[str, np.ndarray]:
    """Where the two starting arrows (1,0) and (0,1) end up — the columns of ``M``."""
    M = np.asarray(M, dtype=float)
    return {"right arrow (1, 0)": M[:, 0], "up arrow (0, 1)": M[:, 1]}


def area_change(M) -> float:
    """How much a shape's area is multiplied by. Negative means the grid was flipped over.

    Zero is the interesting case: the grid has been squashed onto a line, a whole
    dimension has been thrown away, and nothing can undo it.
    """
    return float(np.linalg.det(np.asarray(M, dtype=float)))


def grid_lines(limit: float = 3.0, step: float = 0.5, points_per_line: int = 60):
    """The plain background grid, as a list of lines each shaped ``(points, 2)``.

    Every line is stored as many points rather than two endpoints, because once a squish
    is involved a straight line stops being straight and we need to see the bend.
    """
    ticks = np.arange(-limit, limit + step / 2, step)
    spread = np.linspace(-limit, limit, points_per_line)
    lines = []
    for t in ticks:
        lines.append(np.column_stack([spread, np.full_like(spread, t)]))
        lines.append(np.column_stack([np.full_like(spread, t), spread]))
    return lines


def apply_matrix(points, M) -> np.ndarray:
    """Move a set of points with a matrix."""
    return np.asarray(points, dtype=float) @ np.asarray(M, dtype=float).T


def transform_grid(M, limit: float = 3.0, step: float = 0.5):
    """The grid after ``M`` has moved it. Straight lines stay straight."""
    return [apply_matrix(line, M) for line in grid_lines(limit, step)]


def transform_grid_with_squish(A, B=None, squish: str | None = None, limit: float = 3.0, step: float = 0.5):
    """Apply ``A``, then optionally a squish, then optionally ``B``.

    With ``squish=None`` the result is still perfectly straight — and it is identical to
    what the single matrix ``B @ A`` would have done. With a squish in the middle the
    grid lines curve, and no single matrix can reproduce it.
    """
    lines = grid_lines(limit, step)
    out = []
    for line in lines:
        moved = apply_matrix(line, A)
        if squish is not None:
            moved = ACTIVATIONS[squish][0](moved)
        if B is not None:
            moved = apply_matrix(moved, B)
        out.append(moved)
    return out


def chained_equals_single(A, B) -> tuple[np.ndarray, float]:
    """Check that doing ``A`` then ``B`` is the same as doing the one matrix ``B @ A``.

    Returns the combined matrix and the largest disagreement over a cloud of test points.
    That number comes out as zero, or as a rounding error around 1e-15 — the two really
    are the same transformation, not merely similar ones.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    combined = B @ A

    rng = np.random.default_rng(0)
    test = rng.uniform(-3, 3, size=(200, 2))
    two_steps = apply_matrix(apply_matrix(test, A), B)
    one_step = apply_matrix(test, combined)
    return combined, float(np.abs(two_steps - one_step).max())


def squish_breaks_the_pattern(A, B, squish: str = "tanh") -> float:
    """How far apart the two-step-with-a-squish result is from any single matrix.

    Compare this with the number from :func:`chained_equals_single`. Without the squish
    the gap is rounding error; with it, the gap is large and real.
    """
    rng = np.random.default_rng(0)
    test = rng.uniform(-3, 3, size=(200, 2))

    with_squish = apply_matrix(ACTIVATIONS[squish][0](apply_matrix(test, A)), B)
    best_single = apply_matrix(test, np.asarray(B, dtype=float) @ np.asarray(A, dtype=float))
    return float(np.abs(with_squish - best_single).max())


# ---------------------------------------------------------------------------
# Shapes worth watching get transformed
# ---------------------------------------------------------------------------


def house(n_per_edge: int = 24) -> np.ndarray:
    """A little house outline. A shape with a clear top and a clear left tells you
    instantly whether the grid was rotated, flipped, or sheared — a blob would not."""
    corners = [
        (-1.0, -1.0), (1.0, -1.0), (1.0, 0.6), (0.0, 1.5), (-1.0, 0.6), (-1.0, -1.0),
        (-0.35, -1.0), (-0.35, -0.1), (0.3, -0.1), (0.3, -1.0),
    ]
    points = []
    for start, end in zip(corners[:-1], corners[1:]):
        t = np.linspace(0, 1, n_per_edge, endpoint=False).reshape(-1, 1)
        points.append(np.array(start) + t * (np.array(end) - np.array(start)))
    return np.vstack(points)


def unit_square() -> np.ndarray:
    """The corners of the 1x1 square, used to show what happens to area."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])


# ---------------------------------------------------------------------------
# Going from three dimensions down to two
# ---------------------------------------------------------------------------


def tilted_cloud(n: int = 300, seed: int = 0) -> np.ndarray:
    """A flat-ish, tilted pancake of points in 3D.

    It is genuinely three-dimensional, but nearly all of its spread lives in two
    directions — so a well-chosen shadow keeps almost everything, and a badly chosen one
    throws it away. That contrast is the point.
    """
    rng = np.random.default_rng(seed)
    cloud = rng.normal(size=(n, 3)) * np.array([2.4, 1.1, 0.18])

    tilt = np.array(
        [
            [0.86, -0.35, 0.37],
            [0.44, 0.83, -0.34],
            [-0.26, 0.44, 0.86],
        ]
    )
    return cloud @ tilt.T


def shadow_on_plane(points3d, angle_a_deg: float, angle_b_deg: float) -> np.ndarray:
    """Cast a 3D cloud onto a flat plane chosen by two angles.

    The two angles pick the direction we are looking from; the plane is whatever is at
    right angles to that. Turning the angles turns the shadow — exactly like rotating
    your hand in front of a lamp.
    """
    a = np.radians(angle_a_deg)
    b = np.radians(angle_b_deg)

    view = np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)])

    # Any two directions at right angles to the view direction will do as plane axes.
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(view, helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    axis1 = np.cross(view, helper)
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = np.cross(view, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    return np.asarray(points3d, dtype=float) @ np.column_stack([axis1, axis2])


def spread_of(points) -> float:
    """Total spread — how much the points differ from each other after being flattened.

    A shadow that keeps the points spread out has kept the information. A shadow that
    piles them on top of each other has thrown it away.
    """
    points = np.asarray(points, dtype=float)
    return float(points.var(axis=0).sum())


def best_shadow_spread(points3d) -> float:
    """The most spread any flat shadow can keep. This is what PCA finds in Chapter 21."""
    points = np.asarray(points3d, dtype=float)
    centred = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    variances = (singular_values**2) / len(points)
    return float(variances[0] + variances[1])
