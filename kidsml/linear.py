"""Small linear-model helpers for Part 1.

These functions keep the chapter pages and notebooks in agreement. They are deliberately
plain NumPy: a line has weights, a bias, and a score telling us how wrong it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_line(x, w: float, b: float) -> np.ndarray:
    """Predict ``y`` from one feature with ``w*x + b``."""
    return float(w) * np.asarray(x, dtype=float) + float(b)


def squared_error_table(x, y, w: float, b: float) -> pd.DataFrame:
    """A pencil-friendly table for one candidate regression line."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = predict_line(x, w, b)
    mistake = y - pred
    return pd.DataFrame(
        {
            "x": x,
            "actual": y,
            "prediction": np.round(pred, 2),
            "mistake": np.round(mistake, 2),
            "mistake²": np.round(mistake ** 2, 2),
        }
    )


def mse_for_line(x, y, w: float, b: float) -> float:
    """Mean squared error for one line."""
    pred = predict_line(x, w, b)
    return float(np.mean((pred - np.asarray(y, dtype=float)) ** 2))


def gradient_descent_line(x, y, w: float = 0.0, b: float = 0.0, lr: float = 0.01, steps: int = 80):
    """Walk downhill on mean squared error for ``y = w*x + b``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ws, bs, losses = [], [], []
    for _ in range(steps):
        pred = w * x + b
        losses.append(float(np.mean((pred - y) ** 2)))
        ws.append(float(w))
        bs.append(float(b))
        w = w - lr * float(np.mean(2 * (pred - y) * x))
        b = b - lr * float(np.mean(2 * (pred - y)))
    return {"w": np.array(ws), "b": np.array(bs), "loss": np.array(losses)}


def score_line(X, w1: float, w2: float, b: float) -> np.ndarray:
    """The raw score ``w1*x1 + w2*x2 + b`` for a 2D line."""
    X = np.asarray(X, dtype=float)
    return float(w1) * X[:, 0] + float(w2) * X[:, 1] + float(b)


def predict_side(X, w1: float, w2: float, b: float) -> np.ndarray:
    """Class 1 on the positive side of the line, else class 0."""
    return (score_line(X, w1, w2, b) > 0).astype(int)


def mistake_count(X, y, w1: float, w2: float, b: float) -> int:
    """How many labels the hard line gets wrong."""
    return int((predict_side(X, w1, w2, b) != np.asarray(y, dtype=int)).sum())


def perceptron_history(X, y, w=(0.0, 0.0), b: float = 0.0, steps: int = 30, lr: float = 1.0):
    """Run the perceptron rule and keep every line it draws."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    rows = []
    for step in range(steps):
        mistakes = predict_side(X, w[0], w[1], b) != y
        rows.append({"step": step, "w1": w[0], "w2": w[1], "b": b, "mistakes": int(mistakes.sum())})
        if not mistakes.any():
            break
        i = int(np.flatnonzero(mistakes)[0])
        direction = 1.0 if y[i] == 1 else -1.0
        w = w + lr * direction * X[i]
        b = b + lr * direction
    return pd.DataFrame(rows)


def sigmoid(z):
    """Squash any number into 0..1."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def logistic_proba(X, w1: float, w2: float, b: float) -> np.ndarray:
    """Probability of class 1 from a straight-line score."""
    return sigmoid(score_line(X, w1, w2, b))
