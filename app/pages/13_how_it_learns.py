"""Chapter 13 · How a Neuron Learns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml.datasets import toy_shape
from kidsml.nn_numpy import MLP, Neuron, numeric_gradient
from kidsml.plots import decision_boundary, loss_curve

lesson.begin(13)


@lesson.step("The neuron moves its own sliders", beat="hook")
def _():
    lesson.say(
        """
In Chapter 12 you grabbed the sliders yourself: try a number, watch the mistakes splat
onto the graph, then try a better number.

Now the neuron gets its own tiny steering wheel! The word **gradient** means two things at
once: how much this learned number matters for the mistake, and which way points downhill.
"""
    )
    lesson.mermaid(
        """
graph LR
    X[inputs and weights] --> Z[z score]
    Z --> O[output]
    O --> L[loss]
    L -. blame .-> O
    O -. blame .-> Z
    Z -. blame .-> X
""",
        height=270,
    )
    lesson.look_for("solid arrows for the prediction zooming forward, dotted arrows for blame marching backward.")


@lesson.step("One training step by hand", beat="byhand")
def _():
    lesson.say(
        """
One point: **x = (1, 2)**, answer **1**. Start with **w1 = 0, w2 = 0, b = 0**.

We will compute the first training step with every number on the table. The loss is squared
error, so when the output is too low, `dL/dout` is negative: the arrow says push upward.
"""
    )
    rows = pd.DataFrame(
        [
            ["z", "0*1 + 0*2 + 0", 0.0],
            ["output", "sigmoid(0)", 0.5],
            ["loss", "(0.5 - 1)^2", 0.25],
            ["dL/dout", "2*(0.5 - 1)", -1.0],
            ["sigmoid slope", "at z = 0", 0.25],
            ["dL/dz", "-1 * 0.25", -0.25],
            ["dw1", "-0.25 * x1 = -0.25 * 1", -0.25],
            ["dw2", "-0.25 * x2 = -0.25 * 2", -0.5],
            ["db", "-0.25 * 1", -0.25],
        ],
        columns=["piece", "working", "value"],
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    lesson.look_for("the gradient signs. Negative means raising that number would lower the loss, like finding the downhill edge of a ramp.")


@lesson.step("Read the chain backward", beat="byhand")
def _():
    lesson.mermaid(
        """
graph LR
    W[w1] -->|x1 = 1| Z[z]
    Z -->|slope 0.25| O[output]
    O -->|2(out-y) = -1| L[loss]
""",
        height=220,
    )
    lesson.look_for("the three links in the chain. The chain rule snaps together one small effect after another.")
    lesson.say(
        """
Read backward: `dL/dw1 = -1 * 0.25 * 1 = -0.25`. Three links, one tug.

With **lr = 0.5**, subtract the gradient: `w1 = 0 - 0.5*(-0.25) = 0.125`,
`w2 = 0 - 0.5*(-0.5) = 0.25`, and `b = 0 - 0.5*(-0.25) = 0.125`. One nudge, and the numbers move!
"""
    )
    lesson.aha("Subtracting the gradient walks downhill. If the gradient is negative, subtracting it moves the number up — weird sentence, correct move!")


@lesson.step("Predict the slow check", beat="seeit")
def _():
    guess = lesson.predict(
        "We measure gradients two ways: tiny nudges and backprop. What should happen if backprop is right?",
        ["They disagree wildly", "They match to many decimal places", "Only the bias matches"],
        correct=1,
        why="The slow experiment and the fast formula are measuring the same slope from opposite ends of the tunnel.",
        key="ch12_gradient_match",
    )
    if guess is None:
        return

    X_small = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 1.0]])
    y_small = np.array([1.0, 0.0, 1.0])
    model = MLP([2, 1], activation="sigmoid", seed=0)
    model.Ws[0][:] = np.array([[0.2], [-0.1]])
    model.bs[0][:] = 0.05
    fast_W, fast_b, _ = model.gradients(X_small, y_small)
    slow_W, slow_b = numeric_gradient(model, X_small, y_small)
    proof = pd.DataFrame(
        {
            "piece": ["w1", "w2", "b"],
            "slow numeric gradient": [slow_W[0][0, 0], slow_W[0][1, 0], slow_b[0][0]],
            "fast backprop gradient": [fast_W[0][0, 0], fast_W[0][1, 0], fast_b[0][0]],
        }
    )
    st.dataframe(proof.round(12), hide_index=True)
    lesson.look_for("matching columns. The largest difference is a speck because both routes found the same slopes.")
    st.success(f"Largest difference: {np.max(np.abs(proof.iloc[:, 1] - proof.iloc[:, 2])):.2e}")
    lesson.jargon("gradient", "A number that says how the loss changes if one learned number is nudged upward.")


@st.cache_data(show_spinner=False)
def train_path(lr: float, seed: int, steps: int = 180):
    X, y = toy_shape("blobs", n=160, noise=0.25, seed=3)
    rng = np.random.default_rng(seed)
    n = Neuron(w=rng.normal(0, 0.6, size=2), b=0.0, activation="sigmoid")
    ws, bs, losses = [], [], []
    for _ in range(steps):
        losses.append(n.step(X, y, lr=lr))
        ws.append(n.w.copy())
        bs.append(n.b)
    return X, y, np.array(ws), np.array(bs), np.array(losses)


@lesson.step("Predict a too-large step", beat="play")
def _():
    guess = lesson.predict(
        "What does a too-large learning rate do?",
        ["It always learns faster", "It can jump past good places and explode", "It freezes the weights"],
        correct=1,
        why="Each bad jump lands on a new part of the hill, so the next gradient is measured from a worse place.",
        key="ch12_large_lr",
    )
    if guess is None:
        return

    lesson.say("Try a tiny learning rate, a middle one, and a huge one. The learning rate is the boot size for every downhill step.")
    knobs, picture = lesson.controls()
    with knobs:
        lr = st.slider("Learning rate", 0.0, 8.0, 0.8, 0.1, key="ch12_lr")
        seed = st.slider("Random start", 0, 10, 2, 1, key="ch12_seed")
    X, y, ws, bs, losses = train_path(lr, seed)
    current = Neuron(w=ws[-1], b=float(bs[-1]), activation="sigmoid")
    with picture:
        fig, ax = lesson.figure(5.1, 4.3)
        decision_boundary(lambda G: current.forward(G), X, y, ax=ax, steps=180, title="Boundary after training")
        lesson.show(fig)
    lesson.look_for("whether the final boundary lands cleanly between the blobs or flings itself into nonsense weights.")


@lesson.step("Watch the loss while it learns", beat="play")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        lr = st.slider("Loss learning rate", 0.0, 8.0, 0.8, 0.1, key="ch12_loss_lr")
        seed = st.slider("Loss random start", 0, 10, 2, 1, key="ch12_loss_seed")
    _, _, _, _, losses = train_path(lr, seed)
    with picture:
        fig, ax = lesson.figure(5.1, 4.3)
        loss_curve(losses, ax=ax, title="Loss while it learns")
        lesson.show(fig)
    lesson.look_for("smooth falling, slow crawling, or wild spikes. Those are the three learning-rate weather patterns.")


@lesson.step("The weights walk", beat="play")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        lr = st.slider("Walk learning rate", 0.0, 8.0, 0.8, 0.1, key="ch12_walk_lr")
        seed = st.slider("Walk random start", 0, 10, 2, 1, key="ch12_walk_seed")
    _, _, ws, _, _ = train_path(lr, seed)
    with picture:
        fig, ax = lesson.figure(5.8, 4.2)
        ax.plot(ws[:, 0], ws[:, 1], color="#10B981", marker="o", markersize=2)
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        ax.set_title("The weights walk across the loss valley")
        lesson.show(fig)
    lesson.look_for("whether the walk crawls, settles, or ricochets around the valley walls.")
    lesson.careful("Downhill finds a bottom it can reach from its starting place. On awkward problems, a different start can slide into a different bowl.")


@lesson.step("Learning cannot fix the wrong shape", beat="forreal")
def _():
    lesson.say(
        """
Here is the Chapter 12 wall with learning switched on. XOR still has the wrong shape
for one neuron, so training can lower loss without solving the pattern.
"""
    )
    X_twist, y_twist = toy_shape("xor", n=160, noise=0.05, seed=5)
    starts = []
    for s in [1, 8]:
        rng = np.random.default_rng(s)
        n = Neuron(w=rng.normal(0, 1.0, size=2), b=0.0, activation="sigmoid")
        losses_s = n.fit(X_twist, y_twist, lr=0.7, epochs=500)
        starts.append({"start": s, "final loss": losses_s[-1], "mistakes": int((n.predict(X_twist) != y_twist).sum()), "w1": n.w[0], "w2": n.w[1], "b": n.b})
    st.dataframe(pd.DataFrame(starts).round(3), hide_index=True)
    lesson.look_for("mistakes that remain after training. The gradients are steering a model that owns one straight boundary, not a magic rubber fence.")
    lesson.say("Chapter 14 changes the model, not the downhill idea. Same hill-walking engine, sharper vehicle!")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go tune the step size", beat="challenge")
def _():
    lesson.say(
        """
1. **Find the biggest safe step, no cap.** Raise the learning rate until the loss starts bouncing off the walls.
2. **Break it later.** Find a rate where the first few steps improve, then the curve gets worse.
3. **Set lr to zero.** Explain why the map is not enough without a step.
4. **Explain the bumps.** The loss is measured after jumps, not drawn by a smooth pen.
"""
    )

    lesson.kid_corner(
        "Play beanbag toss at a target. If your throw lands short, toss a bit harder next time. If it sails over the bucket, ease off. How much you change your throw is the learning rate — change it too much and the beanbag keeps flying past the target."
    )


lesson.finish()
