"""Chapter 15 · Same Thing, in PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml import torch_bits as tb
from kidsml.datasets import toy_shape, xor_exact
from kidsml.nn_numpy import MLP
from kidsml.plots import decision_boundary, loss_curve

lesson.begin(15)


@st.cache_data(show_spinner=False)
def gradient_proof():
    X, y = xor_exact()
    numpy_net = MLP([2, 3, 1], activation="tanh", seed=2)
    torch_net = tb.mlp([2, 3, 1], activation="tanh", seed=0)
    tb.copy_from_numpy(torch_net, numpy_net)
    np_dWs, np_dbs, np_loss = numpy_net.gradients(X, y)
    th_dWs, th_dbs, th_loss = tb.gradients(torch_net, X, y)
    rows = []
    for layer in range(len(np_dWs)):
        rows.append({"piece": f"W{layer}", "largest difference": float(np.max(np.abs(np_dWs[layer] - th_dWs[layer])))})
        rows.append({"piece": f"b{layer}", "largest difference": float(np.max(np.abs(np_dbs[layer] - th_dbs[layer])))})
    biggest = max(r["largest difference"] for r in rows)
    assert biggest < 1e-6
    return pd.DataFrame(rows), float(np_loss), float(th_loss), biggest


@st.cache_resource(show_spinner=False)
def train_torch(shape: str, hidden: int, lr: float, seed: int):
    X, y = toy_shape(shape, n=180, noise=0.18, seed=seed)
    model = tb.mlp([2, hidden, 1], activation="tanh", seed=seed)
    result = tb.train(model, X, y, epochs=450, lr=lr)
    return X, y, model, result["losses"], result["seconds"]


@lesson.step("The names changed", beat="hook")
def _():
    lesson.say(
        """
Everything so far was NumPy you could read top to bottom. PyTorch is the grown-up tool,
but it is not a new kind of thinking.

The promise is bigger: tensors remember how they were made.
"""
    )
    lesson.mermaid(
        """
graph LR
    A[NumPy arrays] --> B[torch tensors]
    C[linear layer] --> D[nn.Linear]
    E[manual gradients] --> F[loss.backward]
""",
        height=230,
    )
    lesson.look_for("the one-for-one swaps: arrays become tensors, layers become modules, and manual gradients become backward.")
    lesson.say("If PyTorch remembers the forward recipe, it can walk that recipe backward and fill in gradients for every weight. Chapter 12, at framework speed.")


@lesson.step("A translation dictionary", beat="byhand")
def _():
    guess = lesson.predict(
        "When NumPy code becomes PyTorch code, which part of the learning idea changes?",
        ["The weighted sums", "The squishes", "The names and bookkeeping"],
        correct=2,
        why="PyTorch still needs weighted sums, squishes, loss, and update steps. It handles the bookkeeping for us.",
        key="ch15_translation",
    )
    if guess is None:
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.code(
            """# our NumPy network
net = MLP([2, 3, 1], activation='tanh')
out = net.forward(X)
loss = mse(out, y)
net.step(X, y, lr=0.2)""",
            language="python",
        )
    with col_b:
        st.code(
            """# the PyTorch version
net = nn.Sequential(
    nn.Linear(2, 3), nn.Tanh(),
    nn.Linear(3, 1), nn.Sigmoid(),
)
loss = nn.MSELoss()(net(X), y)
loss.backward()
optimizer.step()""",
            language="python",
        )
    st.dataframe(
        pd.DataFrame(
            {
                "our name": ["a @ W + b", "tanh", "mse", "nudge loop"],
                "PyTorch name": ["nn.Linear", "nn.Tanh", "nn.MSELoss", "optimizer.step"],
            }
        ),
        hide_index=True,
    )
    lesson.look_for("the row-by-row match. Nothing in PyTorch skips the weighted sums, squishes, loss, or step.")


@lesson.step("Tensors keep the recipe", beat="seeit")
def _():
    lesson.say(
        """
A tensor is an array with a notebook attached. When `requires_grad` is on, the notebook
records which operations made the tensor, so `backward()` can retrace them.
"""
    )
    st.code(
        """
x = torch.tensor([2.0], requires_grad=True)
y = x * x + 3
y.backward()
x.grad      # 4, because the slope of x² at x = 2 is 4
""",
        language="python",
    )
    lesson.look_for("the `.grad` bucket. `backward()` filled it by walking the recipe in reverse.")
    lesson.jargon("tensor", "A NumPy-like array that can remember the operations that created it, so gradients can be traced backward.")
    lesson.say(
        """
Those gradient buckets **accumulate**. PyTorch adds new gradients to whatever is already
there because some advanced training loops add blame from several mini-batches before stepping.
"""
    )
    lesson.careful("For our loop, old blame would be stale, so `zero_grad()` clears the buckets first.")


@lesson.step("Will the gradients match?", beat="play")
def _():
    lesson.say("We copy our exact weights into a PyTorch model, run the same XOR points, and compare every gradient.")
    guess = lesson.predict(
        "What do you expect the NumPy and PyTorch gradients to do?",
        ["Agree to tiny rounding error", "Point roughly the same way", "Disagree completely"],
        correct=0,
        why="Both systems run the same forward pass and the same chain rule, so only floating-point dust should remain.",
        key="ch15_gradient_match",
    )
    if guess is None:
        return

    proof, np_loss, th_loss, biggest = gradient_proof()
    st.dataframe(proof, hide_index=True)
    st.success(f"NumPy loss {np_loss:.12f} · PyTorch loss {th_loss:.12f} · biggest gradient difference {biggest:.2e}")
    lesson.look_for("the difference column. Matching gradients mean the next update step is the same step.")


@lesson.step("The real training loop", beat="forreal")
def _():
    lesson.say("Here is the PyTorch version training on a toy shape. The code is shorter because PyTorch handles parameters, tracing, and the step.")
    st.code(
        """
model = nn.Sequential(
    nn.Linear(2, hidden), nn.Tanh(),
    nn.Linear(hidden, 1), nn.Sigmoid(),
)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(450):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(X), y)
    loss.backward()
    optimizer.step()
""",
        language="python",
    )
    lesson.look_for("the same four moves: forward pass, loss, backward gradients, downhill step.")


@lesson.step("Move the PyTorch knobs", beat="forreal")
def _():
    knobs, picture = lesson.controls()
    with knobs:
        shape = st.selectbox("Dataset", ["moons", "xor", "circles"], index=0, key="ch15_shape")
        hidden = st.slider("Hidden neurons", 2, 8, 3, key="ch15_hidden")
        lr = st.slider("PyTorch learning rate", 0.02, 1.0, 0.25, 0.02, key="ch15_lr")
    with picture:
        X, y, torch_model, losses, seconds = train_torch(shape, hidden, lr, 5)
        cols = st.columns(2)
        with cols[0]:
            fig, ax = lesson.figure(5.2, 4.3)
            decision_boundary(lambda G: tb.predict_proba(torch_model, G), X, y, ax=ax, steps=170, title="PyTorch boundary")
            lesson.show(fig)
        with cols[1]:
            fig, ax = lesson.figure(5.2, 4.3)
            loss_curve(losses, ax=ax, title=f"PyTorch trained in {seconds:.2f}s")
            lesson.show(fig)
    lesson.look_for("the loss curve and boundary moving together as the same Part 3 machine learns.")


@lesson.step("Why use the grown-up tool?", beat="forreal")
def _():
    guess = lesson.predict(
        "At this toy size, what is the best reason to use PyTorch?",
        ["It changes the math", "It scales the bookkeeping", "It removes data problems"],
        correct=1,
        why="The math is the same. PyTorch becomes valuable when the model and data are much bigger, like the image chapters coming next.",
        key="ch15_why_torch",
    )
    if guess is None:
        return
    lesson.aha("At this toy size, NumPy is fine. PyTorch starts to matter when the model and data get much bigger.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Add a layer.** Change `[2, 3, 1]` to `[2, 3, 3, 1]`.
2. **Forget zero_grad.** In a notebook, remove it and watch old blame pile up.
3. **Try Adam.** Change `optimizer='adam'` in `torch_bits.train` and compare curves.
4. **Check again.** Copy weights from NumPy and make sure the gradient proof still passes.
"""
    )
    lesson.kid_corner("PyTorch keeps footprints. Then it walks backward to see who stepped in the mud.")


lesson.finish()
