"""Tiny PyTorch helpers for the neural-network chapters.

The API mirrors ``kidsml.nn_numpy`` on purpose. Chapter 17 can put the two side by side
and show that PyTorch is doing the same forward pass and the same backward pass.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == 'tanh':
        return nn.Tanh()
    if name == 'relu':
        return nn.ReLU()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'identity':
        return nn.Identity()
    raise ValueError(f'unknown activation {name!r}')


def mlp(sizes, activation: str = 'tanh', out_activation: str = 'sigmoid', seed: int = 0) -> nn.Sequential:
    """Build a small ``nn.Sequential`` MLP matching ``kidsml.nn_numpy.MLP``."""
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        last = i == len(sizes) - 2
        layers.append(_activation(out_activation if last else activation))
    return nn.Sequential(*layers).double()


def linear_layers(model: nn.Module) -> list[nn.Linear]:
    """The learnable layers, in order."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def copy_from_numpy(torch_model: nn.Module, numpy_model) -> nn.Module:
    """Copy ``kidsml.nn_numpy.MLP`` weights into an equivalent PyTorch model."""
    with torch.no_grad():
        for layer, W, b in zip(linear_layers(torch_model), numpy_model.Ws, numpy_model.bs):
            layer.weight.copy_(torch.as_tensor(W.T, dtype=torch.float64))
            layer.bias.copy_(torch.as_tensor(b, dtype=torch.float64))
    return torch_model


def tensors(X, y=None):
    """NumPy arrays to float64 tensors, with labels shaped like network outputs."""
    X_t = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float64)
    if y is None:
        return X_t
    y_t = torch.as_tensor(np.asarray(y, dtype=float).reshape(-1, 1), dtype=torch.float64)
    return X_t, y_t


def train(model: nn.Module, X, y, epochs: int = 400, lr: float = 0.2, optimizer: str = 'sgd') -> dict[str, object]:
    """Train with MSE and return losses plus elapsed wall-clock seconds."""
    X_t, y_t = tensors(X, y)
    if optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return {'losses': np.array(losses), 'seconds': time.perf_counter() - start, 'model': model}


def predict_proba(model: nn.Module, X) -> np.ndarray:
    """Return one probability per row as a NumPy array."""
    with torch.no_grad():
        out = model(tensors(X))
    return out.detach().cpu().numpy().reshape(-1)


def gradients(model: nn.Module, X, y):
    """Run one backward pass and return gradients shaped like ``nn_numpy.MLP``."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    X_t, y_t = tensors(X, y)
    loss = nn.MSELoss()(model(X_t), y_t)
    loss.backward()
    dWs, dbs = [], []
    for layer in linear_layers(model):
        dWs.append(layer.weight.grad.detach().cpu().numpy().T.copy())
        dbs.append(layer.bias.grad.detach().cpu().numpy().copy())
    return dWs, dbs, float(loss.detach())


def predict_labels(model: nn.Module, X, threshold: float = 0.5) -> np.ndarray:
    """Hard 0/1 answers from a PyTorch model."""
    return (predict_proba(model, X) >= threshold).astype(int)
