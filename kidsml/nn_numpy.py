"""Neural networks written out by hand, in NumPy.

Nothing here is hidden and nothing is imported from a deep-learning framework. If you
read this file top to bottom you have read *all* of the maths a neural network does.

The layout mirrors the chapters:

* :func:`sigmoid`, :func:`relu`, :func:`tanh_`  — the "squish" functions (Ch 11)
* :class:`Neuron`                              — one neuron, forward and backward (Ch 11-12)
* :class:`MLP`                                 — layers of neurons (Ch 13-14)

Shapes convention used everywhere: ``X`` is ``(n_samples, n_features)`` and a layer's
weight matrix ``W`` is ``(n_inputs, n_outputs)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Squish functions ("activations") and their slopes
# ---------------------------------------------------------------------------


def sigmoid(z):
    """Squash any number into the range 0..1.

    Very negative -> near 0. Zero -> exactly 0.5. Very positive -> near 1.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def sigmoid_slope(z):
    """How steep the sigmoid is at ``z``. Steepest in the middle, flat at the ends."""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    """Keep positive numbers, flatten negative ones to zero. A hinge."""
    return np.maximum(0.0, z)


def relu_slope(z):
    """The slope of ReLU: 1 where the input was positive, 0 where it was not."""
    return (np.asarray(z) > 0).astype(float)


def tanh_(z):
    """Like sigmoid, but squashes into -1..1 instead of 0..1."""
    return np.tanh(np.clip(z, -60, 60))


def tanh_slope(z):
    """The slope of tanh."""
    t = tanh_(z)
    return 1.0 - t * t


def identity(z):
    """No squish at all — used on the output when we predict a plain number."""
    return z


def identity_slope(z):
    """The slope of doing nothing is 1."""
    return np.ones_like(np.asarray(z, dtype=float))


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_slope),
    "relu": (relu, relu_slope),
    "tanh": (tanh_, tanh_slope),
    "identity": (identity, identity_slope),
}

ACTIVATION_BLURB = {
    "sigmoid": "An S-curve. Smooth, and it never leaves 0..1.",
    "relu": "A hinge. Flat then straight. Boundaries end up looking like folded paper.",
    "tanh": "An S-curve centred on zero. Smooth, curvy boundaries.",
    "identity": "No squish. The neuron stays a plain straight line.",
}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def mse(pred, target):
    """Mean squared error — the average of (mistake)^2. For predicting numbers."""
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))


def mse_slope(pred, target):
    """How the MSE changes when a prediction changes."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return 2.0 * (pred - target) / pred.size


def log_loss(prob, target, eps: float = 1e-9):
    """How surprised we are by the true answer. Small when confident *and* right."""
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    t = np.asarray(target, dtype=float)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def log_loss_slope(prob, target, eps: float = 1e-9):
    """How the log loss changes when the predicted probability changes."""
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    t = np.asarray(target, dtype=float)
    return (p - t) / (p * (1 - p)) / p.size


# ---------------------------------------------------------------------------
# One neuron
# ---------------------------------------------------------------------------


@dataclass
class Neuron:
    """A single neuron: multiply, add, squish.

    ``output = squish(w1*x1 + w2*x2 + ... + b)``

    That is the whole thing. Chapter 11 builds this by hand; chapter 12 teaches it to
    learn with :meth:`step`.
    """

    n_inputs: int = 2
    activation: str = "sigmoid"
    w: np.ndarray = field(default=None)
    b: float = 0.0

    def __post_init__(self):
        if self.w is None:
            self.w = np.zeros(self.n_inputs)
        self.w = np.asarray(self.w, dtype=float)
        self.n_inputs = self.w.size

    @property
    def _act(self):
        return ACTIVATIONS[self.activation]

    def raw(self, X):
        """The weighted sum *before* the squish. Often written ``z``."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def forward(self, X):
        """The neuron's answer for every row of ``X``."""
        return self._act[0](self.raw(X))

    def predict(self, X, threshold: float = 0.5):
        """Turn the neuron's answer into a hard 0 or 1."""
        return (self.forward(X) >= threshold).astype(int)

    def gradients(self, X, y):
        """Work out which way to nudge ``w`` and ``b`` to be less wrong.

        Returns ``(dw, db, loss)``. This is backpropagation for one neuron, written
        out in three lines so the chain rule is visible:

        1. how does the loss change with the output?   ``dL_dout``
        2. how does the output change with the raw sum? ``dout_dz`` (the squish's slope)
        3. how does the raw sum change with each weight? ``x`` itself
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        z = self.raw(X)
        out = self._act[0](z)

        dL_dout = mse_slope(out, y)
        dout_dz = self._act[1](z)
        dL_dz = dL_dout * dout_dz

        dw = X.T @ dL_dz
        db = float(dL_dz.sum())
        return dw, db, mse(out, y)

    def step(self, X, y, lr: float = 0.5):
        """One nudge downhill. Returns the loss *before* the nudge."""
        dw, db, loss = self.gradients(X, y)
        self.w = self.w - lr * dw
        self.b = self.b - lr * db
        return loss

    def fit(self, X, y, lr: float = 0.5, epochs: int = 400):
        """Nudge over and over. Returns the list of losses so you can plot the curve."""
        return [self.step(X, y, lr) for _ in range(epochs)]

    def __repr__(self):
        ws = ", ".join(f"{v:+.2f}" for v in self.w)
        return f"Neuron({ws} | b={self.b:+.2f}, squish={self.activation})"


def perceptron_step(w, b, x, y_true, lr: float = 1.0):
    """The original 1958 learning rule — simple enough to run with a pencil.

    Guess with the current line. If the guess is right, change nothing. If it is
    wrong, move the line *towards* the point. Returns ``(w, b, was_wrong)``.
    """
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    guess = 1 if (w @ x + b) > 0 else 0
    if guess == y_true:
        return w, b, False
    direction = 1.0 if y_true == 1 else -1.0
    return w + lr * direction * x, b + lr * direction, True


# ---------------------------------------------------------------------------
# A stack of neurons
# ---------------------------------------------------------------------------


class MLP:
    """A plain feed-forward neural network, written out by hand.

    ``MLP([2, 3, 1])`` means: 2 inputs, a hidden layer of 3 neurons, 1 output neuron.
    That is exactly the network in chapter 13.

    Everything is stored in ``self.Ws`` and ``self.bs`` — ordinary NumPy arrays you can
    print, plot, or edit with a slider.
    """

    def __init__(self, sizes, activation: str = "tanh", out_activation: str = "sigmoid", seed: int = 0):
        self.sizes = list(sizes)
        self.activation = activation
        self.out_activation = out_activation
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None):
        """Start over with fresh random weights."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        self.Ws, self.bs = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            # Small random numbers. If every weight started equal, every neuron in a
            # layer would learn exactly the same thing and the layer would be pointless.
            scale = np.sqrt(1.0 / n_in)
            self.Ws.append(rng.normal(0, scale, size=(n_in, n_out)))
            self.bs.append(np.zeros(n_out))
        return self

    @property
    def n_layers(self) -> int:
        return len(self.Ws)

    def _act_for(self, layer_index: int):
        last = layer_index == self.n_layers - 1
        return ACTIVATIONS[self.out_activation if last else self.activation]

    def forward(self, X, keep: bool = False):
        """Run the network. With ``keep=True`` also return every in-between value.

        The kept values are what chapter 13 draws: each hidden neuron's own output.
        """
        a = np.asarray(X, dtype=float)
        zs, activations = [], [a]
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            a = self._act_for(i)[0](z)
            zs.append(z)
            activations.append(a)
        return (a, zs, activations) if keep else a

    def predict_proba(self, X):
        """The output neuron's number, one per row, flattened."""
        return self.forward(X).ravel()

    def predict(self, X, threshold: float = 0.5):
        """Hard 0/1 answers."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def hidden_outputs(self, X, layer: int = 0):
        """What each neuron in a hidden layer says, for every row of ``X``."""
        _, _, activations = self.forward(X, keep=True)
        return activations[layer + 1]

    def gradients(self, X, y):
        """Backpropagation: push the blame backwards through the layers.

        Returns ``(dWs, dbs, loss)``. The loop is short on purpose — every layer does
        exactly the same three things, just with different numbers.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        out, zs, activations = self.forward(X, keep=True)
        loss = mse(out, y)

        dWs = [None] * self.n_layers
        dbs = [None] * self.n_layers

        # Start at the end: how does the loss change with the final output?
        delta = mse_slope(out, y) * self._act_for(self.n_layers - 1)[1](zs[-1])

        for i in range(self.n_layers - 1, -1, -1):
            dWs[i] = activations[i].T @ delta
            dbs[i] = delta.sum(axis=0)
            if i > 0:
                # Hand the blame back one layer, then through that layer's squish.
                delta = (delta @ self.Ws[i].T) * self._act_for(i - 1)[1](zs[i - 1])

        return dWs, dbs, loss

    def step(self, X, y, lr: float = 0.5, weight_decay: float = 0.0):
        """One nudge downhill for every weight in the network."""
        dWs, dbs, loss = self.gradients(X, y)
        for i in range(self.n_layers):
            if weight_decay:
                dWs[i] = dWs[i] + weight_decay * self.Ws[i]
            self.Ws[i] = self.Ws[i] - lr * dWs[i]
            self.bs[i] = self.bs[i] - lr * dbs[i]
        return loss

    def fit(self, X, y, lr: float = 0.5, epochs: int = 500, weight_decay: float = 0.0, record_every: int = 1):
        """Train, and hand back the loss curve."""
        losses = []
        for e in range(epochs):
            loss = self.step(X, y, lr=lr, weight_decay=weight_decay)
            if e % record_every == 0:
                losses.append(loss)
        return losses

    def accuracy(self, X, y) -> float:
        """Fraction of rows we get right."""
        return float((self.predict(X) == np.asarray(y).ravel()).mean())

    def n_parameters(self) -> int:
        """How many numbers this network has to learn."""
        return sum(W.size for W in self.Ws) + sum(b.size for b in self.bs)

    def describe(self) -> str:
        """A one-line, kid-readable summary."""
        shape = " → ".join(str(s) for s in self.sizes)
        return f"{shape}  ({self.n_parameters()} numbers to learn, squish = {self.activation})"

    def __repr__(self):
        return f"MLP({self.describe()})"


# ---------------------------------------------------------------------------
# A sanity check we can point at
# ---------------------------------------------------------------------------


def numeric_gradient(model: MLP, X, y, eps: float = 1e-6):
    """Estimate the gradients the slow, obvious way: nudge a weight, see what happens.

    Chapter 12 uses this to prove the backprop code is right — and chapter 15 uses the
    same trick to show PyTorch agrees with us.
    """
    dWs = [np.zeros_like(W) for W in model.Ws]
    dbs = [np.zeros_like(b) for b in model.bs]
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    for i in range(model.n_layers):
        for idx in np.ndindex(model.Ws[i].shape):
            original = model.Ws[i][idx]
            model.Ws[i][idx] = original + eps
            up = mse(model.forward(X), y)
            model.Ws[i][idx] = original - eps
            down = mse(model.forward(X), y)
            model.Ws[i][idx] = original
            dWs[i][idx] = (up - down) / (2 * eps)

        for j in range(model.bs[i].size):
            original = model.bs[i][j]
            model.bs[i][j] = original + eps
            up = mse(model.forward(X), y)
            model.bs[i][j] = original - eps
            down = mse(model.forward(X), y)
            model.bs[i][j] = original
            dbs[i][j] = (up - down) / (2 * eps)

    return dWs, dbs
