# %% [markdown]
# # Chapter 11 · Arrows and Grids
#
# ### A matrix is not a box of numbers. It is an instruction for moving space.
#
# *Part 3 · Neural networks*
#
# ---
#
# This notebook is meant to be dragged, not stared at. Run the cells with **Shift + Enter**.
# The widget cells render in JupyterLab, and the first picture appears during tests.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from ipywidgets import Checkbox, Dropdown, FloatSlider, IntSlider

from kidsml.interactive import interact
from plotly.subplots import make_subplots

from kidsml import linalg as la
from kidsml import workbook
from kidsml.nn_numpy import ACTIVATIONS
from kidsml.plots import style_plotly
from kidsml.plots import use_house_style

use_house_style()

BLUE = "#2563EB"
GREEN = "#10B981"
ORANGE = "#F59E0B"
PINK = "#EC4899"
PURPLE = "#7C3AED"
INK = "#0F172A"
GREY = "#CBD5E1"


def setup_axes(ax, limit=4.0, title=""):
    ax.axhline(0, color="#94A3B8", linewidth=1.1)
    ax.axvline(0, color="#94A3B8", linewidth=1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, color="#E2E8F0", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)


def draw_arrow(ax, end, start=(0, 0), color=BLUE, label="", linewidth=3.0):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=linewidth, color=color, shrinkA=0, shrinkB=0))
    if label:
        pos = start + 0.56 * (end - start)
        ax.text(pos[0], pos[1], label, color=color, fontsize=10, weight="bold")


def draw_grid(ax, lines, color=BLUE, alpha=0.85, linewidth=1.0):
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color=color, alpha=alpha, linewidth=linewidth)


def grid_figure(M, title="Matrix as a grid mover", limit=2.5, step=0.5):
    original = la.grid_lines(limit=limit, step=step)
    moved = la.transform_grid(M, limit=limit, step=step)
    house = la.house()
    house_moved = la.apply_matrix(house, M)
    square_moved = la.apply_matrix(la.unit_square(), M)
    landed = la.where_the_arrows_land(M)
    basis = np.vstack([np.zeros(2), landed["right arrow (1, 0)"], landed["up arrow (0, 1)"], house_moved, square_moved])
    bound = max(3.2, float(np.max(np.abs(basis))) * 1.25)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    setup_axes(ax, bound, title)
    draw_grid(ax, original, color=GREY, alpha=0.55, linewidth=0.8)
    draw_grid(ax, moved, color=BLUE, alpha=0.9, linewidth=1.15)
    ax.fill(square_moved[:, 0], square_moved[:, 1], color=ORANGE, alpha=0.25)
    ax.plot(square_moved[:, 0], square_moved[:, 1], color=ORANGE, linewidth=2.0)
    ax.plot(house[:, 0], house[:, 1], color="#94A3B8", linewidth=1.2, linestyle="--", alpha=0.75)
    ax.plot(house_moved[:, 0], house_moved[:, 1], color=INK, linewidth=2.4)
    draw_arrow(ax, landed["right arrow (1, 0)"], color=PINK, label="(1,0)")
    draw_arrow(ax, landed["up arrow (0, 1)"], color=GREEN, label="(0,1)")
    return fig

# %% [markdown]
# ## 🎣 Start here
#
# You may have been told a matrix is a box of numbers. That is like being told a song is
# a box of dots on lines.
#
# Chapter 10 taught you to distrust a shiny score. Now we build the score machine more
# carefully, so there is no hidden magic step when neural networks arrive.
#
# In this chapter a matrix is an **instruction for moving space**. The grid skates, the
# house flips, the arrows land, and the neuron equation in Chapter 12 becomes something
# you can see.
#
# ```mermaid
# graph LR
#     A[An arrow] --> B[A grid full of arrows]
#     B --> C[A matrix moves the grid]
#     C --> D[A squish bends space]
#     D --> E[A neural network can make new shapes]
#     E --> F[Chapter 14 stacks bends into new features]
# ```

# %% [markdown]
# ## ✏️ Work it out
#
# A vector is a little trip: how far, and which way. The vector `[3, 4]` pops open the old
# 3-4-5 triangle.

# %%
v = np.array([3, 4])
vector_facts = pd.DataFrame(
    {
        "thing": ["vector", "length", "angle", "direction arrow"],
        "value": [str(v), f"{la.magnitude(v):.2f}", f"{la.angle_degrees(v):.2f}", str(np.round(la.direction(v), 2))],
    }
)
vector_facts

# %%
fig, ax = plt.subplots(figsize=(5.8, 4.8))
setup_axes(ax, 5.5, "The 3-4-5 arrow")
ax.plot([0, 3, 3], [0, 0, 4], color=GREY, linestyle="--", linewidth=2)
draw_arrow(ax, [3, 4], color=BLUE, label="[3, 4]")
display(Markdown("**Look for:** the dashed legs. Length is `sqrt(3² + 4²) = 5`."))
plt.show()

# %% [markdown]
# Add arrows by walking tip to tail. Scale an arrow by stretching it, shrinking it, or
# flipping it backward.

# %%
a = np.array([2, 1])
b = 1.5 * np.array([1, 2])
fig, ax = plt.subplots(figsize=(5.8, 4.8))
setup_axes(ax, 6.0, "Tip-to-tail addition")
draw_arrow(ax, a, color=BLUE, label="a")
draw_arrow(ax, a + b, start=a, color=ORANGE, label="1.5b")
draw_arrow(ax, a + b, color=GREEN, label="a + 1.5b", linewidth=2.4)
display(Markdown("**Look for:** the green shortcut lands at the same point as the two-arrow walk."))
plt.show()

# %% [markdown]
# The dot product is agreement. Positive means same-way. Zero means right angle. Negative
# means disagreement.
#
# A neuron's weighted sum `w·x` is exactly this: **how much does this input look like the
# thing the neuron is watching for?**

# %%
angles = [-120, -90, 0, 60]
rows = []
watch = np.array([3.0, 0.0])
for angle in angles:
    other = 3 * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    rows.append({"angle": angle, "dot product": round(la.dot(watch, other), 2)})
pd.DataFrame(rows)

# %% [markdown]
# ## 👀 Take a look
#
# The columns of a matrix are where the starter arrows land. Column one is the landing pad
# for `(1, 0)`. Column two is the landing pad for `(0, 1)`. Once those two are known,
# everything else is forced.

# %%
M = la.matrix(1, 1, 0, 1)
la.where_the_arrows_land(M)

# %%
fig = grid_figure(M, "A shear moves every grid point")
display(Markdown("**Look for:** grey before, blue after, and the two coloured landing arrows."))
plt.show()

# %% [markdown]
# > 📖 **Grown-ups call this:** **linear**. Grid lines stay straight and evenly spaced.
# > No secret extra wiggle is allowed.

# %% [markdown]
# ## 🎛️ Your turn
#
# ### The grid mover
#
# Pick a preset, or choose `custom sliders`. The unit square becomes the orange
# parallelogram. The little house makes flips and rotations easy to spot.

# %%
def play_grid(preset, a, b, c, d):
    if preset == "custom sliders":
        M = la.matrix(a, b, c, d)
        blurb = "Your custom matrix."
    else:
        M = la.matrix(*la.PRESETS[preset])
        blurb = la.PRESET_BLURB[preset]
    display(Markdown(f"**{preset}** — {blurb}"))
    landed = la.where_the_arrows_land(M)
    display(pd.DataFrame({"arrow": list(landed), "lands at": [np.round(v, 2) for v in landed.values()]}))
    display(Markdown(f"Area multiplier: **{la.area_change(M):.2f}**"))
    fig = grid_figure(M, "The grid mover")
    plt.show()

interact(
    play_grid,
    preset=Dropdown(options=["custom sliders"] + list(la.PRESETS), value="shear (push the top over)", description="preset"),
    a=FloatSlider(value=1.0, min=-3.0, max=3.0, step=0.1, description="a"),
    b=FloatSlider(value=1.0, min=-3.0, max=3.0, step=0.1, description="b"),
    c=FloatSlider(value=0.0, min=-3.0, max=3.0, step=0.1, description="c"),
    d=FloatSlider(value=1.0, min=-3.0, max=3.0, step=0.1, description="d"),
);

# %% [markdown]
# ### Area and collapse
#
# The determinant is the area multiplier. Negative means the grid flipped over. Zero means
# the square pancaked into a line and a whole dimension is gone.

# %%
collapse = la.matrix(*la.PRESETS["collapse onto a line"])
area_table = pd.DataFrame(
    {
        "matrix": ["stretch", "mirror", "collapse"],
        "determinant": [la.area_change(la.matrix(2, 0, 0, 3)), la.area_change(la.matrix(-1, 0, 0, 1)), la.area_change(collapse)],
        "what it means": ["area becomes 6 times bigger", "same area, flipped over", "area becomes zero"],
    }
)
area_table

# %%
fig = grid_figure(collapse, "Everything lands on one line", limit=1.8)
display(Markdown("**Look for:** many blue grid lines stacked on the same line."))
plt.show()

# %% [markdown]
# ### Projection is a shadow
#
# Projecting one arrow onto another drops a shadow and keeps the part pointing along the chosen direction.

# %%
v = np.array([3.0, 2.0])
onto = np.array([1.0, 0.3])
shadow = la.project_onto(v, onto)
fig, ax = plt.subplots(figsize=(5.8, 4.8))
setup_axes(ax, 4.5, "Projection")
line = np.vstack([-5 * la.direction(onto), 5 * la.direction(onto)])
ax.plot(line[:, 0], line[:, 1], color=GREY, linewidth=2)
draw_arrow(ax, v, color=BLUE, label="arrow")
draw_arrow(ax, shadow, color=ORANGE, label="shadow")
ax.plot([v[0], shadow[0]], [v[1], shadow[1]], color="#94A3B8", linestyle="--")
display(Markdown("**Look for:** the orange shadow is the part the projection keeps."))
plt.show()

# %% [markdown]
# ### 3D → 2D shadows
#
# This is a game: rotate the lamp and keep as much spread as possible. PCA in Chapter 21
# does the same search in one step.

# %%
points3d = la.tilted_cloud(n=260, seed=11)
best = la.best_shadow_spread(points3d)


def shadow_widget(angle_a, angle_b):
    shadow = la.shadow_on_plane(points3d, angle_a, angle_b)
    spread = la.spread_of(shadow)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], subplot_titles=("3D cloud", "2D shadow"))
    fig.add_trace(go.Scatter3d(x=points3d[:, 0], y=points3d[:, 1], z=points3d[:, 2], mode="markers", marker=dict(size=3, opacity=0.75), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=shadow[:, 0], y=shadow[:, 1], mode="markers", marker=dict(size=6, opacity=0.75), showlegend=False), row=1, col=2)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
    fig.update_layout(title=f"Your spread {spread:.2f}; best {best:.2f}")
    style_plotly(fig, height=420).show()

interact(
    shadow_widget,
    angle_a=IntSlider(value=20, min=-180, max=180, step=5, description="spin"),
    angle_b=IntSlider(value=25, min=-80, max=80, step=5, description="tilt"),
);

# %% [markdown]
# ## 💻 In real code
#
# Chaining two matrices is still one matrix. Doing `A`, then `B`, lands in the same place
# as doing `B @ A`. Same footprints!

# %%
A = la.matrix(1.0, 0.6, 0.0, 1.0)
B = la.matrix(0.7, -0.8, 0.8, 0.7)
combined, gap = la.chained_equals_single(A, B)
pd.DataFrame(
    {
        "thing": ["combined matrix B @ A", "largest difference on 200 test points"],
        "value": [str(np.round(combined, 3)), f"{gap:.8f}"],
    }
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, title, lines in zip(
    axes,
    ["after A", "after A then B", "after B @ A"],
    [la.transform_grid(A, limit=2.0), la.transform_grid(B @ A, limit=2.0), la.transform_grid(combined, limit=2.0)],
):
    setup_axes(ax, 4.0, title)
    draw_grid(ax, lines, color=BLUE, alpha=0.9)
display(Markdown("**Look for:** the last two pictures match."))
plt.show()

# %% [markdown]
# That is why stacking linear layers with no activation function buys no new shape power.
# Ten no-squish layers fold down into one matrix wearing a tall hat.

# %%
base = la.matrix(1.05, 0.2, -0.1, 0.95)
combined_10 = np.linalg.matrix_power(base, 10)
combined_10

# %% [markdown]
# Now put a squish in the middle. The last step proved the trap: stack straight moves with
# no squish, and they fuse into one straight move. The squish bends the grid lines, and the
# single-matrix copy fails.
#
# Chapter 12 uses one squish on one neuron. Chapter 14 stacks squished neurons so a hidden
# layer can invent new features.

# %%
A_squish = la.matrix(0.5, -1.25, -0.25, -0.5)
B_squish = la.matrix(1.25, 1.5, 0.0, -1.5)


def squish_widget(use_squish, activation):
    squish = activation if use_squish else None
    lines = la.transform_grid_with_squish(A_squish, B_squish, squish=squish, limit=3.0, step=0.5)
    gap = la.squish_breaks_the_pattern(A_squish, B_squish, activation) if use_squish else la.chained_equals_single(A_squish, B_squish)[1]
    display(Markdown(f"Difference from one plain matrix: **{gap:.2f}**"))
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    setup_axes(ax, 4.6, "The squish bends the grid" if use_squish else "No squish: straight grid")
    draw_grid(ax, lines, color=PURPLE if use_squish else BLUE, alpha=0.9, linewidth=1.2)
    plt.show()

interact(
    squish_widget,
    use_squish=Checkbox(value=True, description="use squish"),
    activation=Dropdown(options=list(ACTIVATIONS), value="tanh", description="activation"),
);

# %% [markdown]
# Chapter 12 writes one neuron as `z = w1*x1 + w2*x2 + b`. That weighted-sum part is the
# same arrow-agreement idea.
# Without the squish, two matrix layers fuse exactly. With `tanh` inserted, they do not.

# %%
x = np.array([1.0, -1.0])
W1 = np.array([[1.0, 2.0], [-0.5, 1.0]])
W2 = np.array([[2.0, -1.0], [0.5, 1.5]])
b = np.array([0.25, -0.75])

z = W1 @ x + b
linear_gap = float(np.max(np.abs(W2 @ (W1 @ x) - (W2 @ W1) @ x)))
squish_gap = float(np.max(np.abs(W2 @ np.tanh(W1 @ x) - (W2 @ W1) @ x)))

pd.DataFrame(
    {
        "quantity": ["z = W1 @ x + b", "no-squish gap", "with tanh gap"],
        "value": [str(np.round(z, 3)), f"{linear_gap:.8f}", f"{squish_gap:.3f}"],
    }
)

# %% [markdown]
# ## 🏆 Go further
#
# Work through the interactive questions, then try the quests below. Move the space with your hands.

# %%
workbook.render(11)

# %% [markdown]
# 1. **Rotate and double.** Find a matrix that spins the house and makes it twice as big.
# 2. **Undo button.** Find two different matrices whose product is the identity, so the
#    second one undoes the first.
# 3. **Upside-down house.** Make the house land on its roof.
# 4. **Worst shadow.** In the 3D shadow game, hunt for the smallest spread.
# 5. **Area bet.** Predict the determinant before checking the area meter.
# 6. **Bonus:** hunt for arrows that keep their direction under a transformation. Grown-ups
#    call those **eigenvectors**.
#
# > 🧸 **Little Kid Corner** — Use a torch and your hand to make shadows on a wall.
# > Stretch a drawing on a balloon. Squash a photo on a screen. Same idea: the object stays
# > one thing, but the space around it gets moved.

# %% [markdown]
# ---
# **Next up:** Chapter 12 · *One Neuron* — where `w·x + b` becomes a visible score with a squish on the end.
