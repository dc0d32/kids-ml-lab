"""Vision helpers for the picture chapters.

Chapter 18 treats an image as a row of numbers. Chapter 19 puts the neighbour
relationships back with a tiny sliding window and a tiny CNN. The functions here
are deliberately small and plain so the notebook code can point straight at them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import warnings

from kidsml.datasets import digits as load_digits
from kidsml.plots import ACCENT, MUTED, show_image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "torchvision"

FASHION_LABELS = (
    "T-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
)

KERNEL_PRESETS: dict[str, np.ndarray] = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
    "blur": np.ones((3, 3), dtype=float) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "vertical edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
    "horizontal edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
}


@dataclass
class DigitMlpResult:
    model: Any
    accuracy: float
    confusion: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    images_train: np.ndarray
    images_test: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


@dataclass
class TorchVisionResult:
    dataset_name: str
    labels: tuple[str, ...]
    image_size: int
    cnn: Any
    mlp: Any
    cnn_accuracy: float
    mlp_accuracy: float
    cnn_params: int
    mlp_params: int
    cnn_predictions: np.ndarray
    mlp_predictions: np.ndarray
    y_test: np.ndarray
    test_images: np.ndarray
    sample_tensor: Any
    elapsed: float
    epochs: int
    train_size: int
    test_size: int


# ---------------------------------------------------------------------------
# Chapter 18: pictures as rows of numbers
# ---------------------------------------------------------------------------


def digit_as_flat_row(image: np.ndarray) -> pd.DataFrame:
    """One 8x8 digit as the 64-number row a plain MLP receives."""
    flat = np.asarray(image, dtype=float).reshape(1, -1)
    columns = [f"p{i:02d}" for i in range(flat.shape[1])]
    return pd.DataFrame(flat.astype(int), columns=columns)


def train_digit_mlp(seed: int = 0) -> DigitMlpResult:
    """Train a small sklearn MLP on the 8x8 digit dataset."""
    X, y, images = load_digits()
    X = X / 16.0
    split = train_test_split(X, y, images, test_size=0.25, random_state=seed, stratify=y)
    X_train, X_test, y_train, y_test, images_train, images_test = split

    model = MLPClassifier(
        hidden_layer_sizes=(48,),
        max_iter=200,
        alpha=1e-4,
        random_state=seed,
        learning_rate_init=0.001,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    accuracy = float((predictions == y_test).mean())
    confusion = confusion_matrix(y_test, predictions, labels=list(range(10)))
    return DigitMlpResult(
        model=model,
        accuracy=accuracy,
        confusion=confusion,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        images_train=images_train,
        images_test=images_test,
        predictions=predictions,
        probabilities=probabilities,
    )


def average_digit_images(images: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The ghostly average 0, average 1, ... average 9."""
    averages = []
    for digit in range(10):
        averages.append(np.asarray(images)[np.asarray(y) == digit].mean(axis=0))
    return np.asarray(averages)


def first_layer_images(model: Any, limit: int = 12) -> np.ndarray:
    """Turn the first hidden layer's 64 input weights into tiny images."""
    weights = np.asarray(model.coefs_[0]).T
    limit = min(limit, len(weights))
    return weights[:limit].reshape(limit, 8, 8)


def misclassified_examples(result: DigitMlpResult, limit: int = 8):
    """Images the MLP got wrong, with true and predicted labels."""
    wrong = np.flatnonzero(result.predictions != result.y_test)
    rows = []
    for idx in wrong[:limit]:
        rows.append((result.images_test[idx], int(result.y_test[idx]), int(result.predictions[idx])))
    return rows


def canvas_to_digit_grid(image_data: Any) -> np.ndarray:
    """Canvas image -> an 8x8 grid scaled like sklearn digits: 0..16.

    The model has only seen neat centred digits. This crop-and-resize step tries to
    match that world, but a messy drawing is still a different kind of data.
    """
    if image_data is None:
        return np.zeros((8, 8), dtype=float)

    arr = np.asarray(image_data)
    if arr.size == 0:
        return np.zeros((8, 8), dtype=float)

    if arr.ndim == 3:
        rgb = arr[:, :, :3].astype(float) / 255.0
        gray = rgb.mean(axis=2)
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3].astype(float) / 255.0
            gray = gray * alpha
    else:
        gray = arr.astype(float)
        if gray.max() > 1:
            gray = gray / 255.0

    return image_to_digit_grid(gray)


def image_to_digit_grid(gray: np.ndarray) -> np.ndarray:
    """Crop, centre, and shrink a white-on-black drawing to 8x8 values."""
    gray = np.asarray(gray, dtype=float)
    if gray.max() > 1:
        gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0)

    rows, cols = np.where(gray > 0.05)
    if len(rows) == 0:
        return np.zeros((8, 8), dtype=float)

    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = gray[top:bottom, left:right]

    h, w = crop.shape
    side = max(h, w)
    pad = max(2, int(side * 0.25))
    square = np.zeros((side + 2 * pad, side + 2 * pad), dtype=float)
    r0 = pad + (side - h) // 2
    c0 = pad + (side - w) // 2
    square[r0:r0 + h, c0:c0 + w] = crop

    img = Image.fromarray(np.uint8(square * 255))
    small = img.resize((8, 8), Image.Resampling.LANCZOS)
    out = np.asarray(small, dtype=float) / 255.0
    if out.max() > 0:
        out = out / out.max()
    return np.clip(out * 16.0, 0.0, 16.0)


def predict_digit_grid(model: Any, grid: np.ndarray) -> tuple[int, np.ndarray]:
    """Predict one hand-made 8x8 grid with the chapter-16 MLP."""
    X = np.asarray(grid, dtype=float).reshape(1, -1) / 16.0
    probabilities = model.predict_proba(X)[0]
    return int(probabilities.argmax()), probabilities


def confidence_table(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"digit": list(range(10)), "confidence": np.asarray(probabilities)})


# ---------------------------------------------------------------------------
# Chapter 19: sliding windows and tiny CNNs
# ---------------------------------------------------------------------------


def convolve2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide a small kernel over an image with plain loops."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    out_h = image.shape[0] - kernel.shape[0] + 1
    out_w = image.shape[1] - kernel.shape[1] + 1
    out = np.zeros((out_h, out_w), dtype=float)
    for r in range(out_h):
        for c in range(out_w):
            patch = image[r:r + kernel.shape[0], c:c + kernel.shape[1]]
            out[r, c] = float((patch * kernel).sum())
    return out


def generated_pattern(size: int = 28) -> np.ndarray:
    """A bigger no-download picture: bright square, stripes, and a diagonal."""
    img = np.zeros((size, size), dtype=float)
    img[:, size // 2:] = 0.35
    img[5:size - 5, 8:size - 8] += 0.35
    for i in range(size):
        img[i, i] = 1.0
        if i + 1 < size:
            img[i, i + 1] = 0.8
    img[::4, :] += 0.15
    return np.clip(img, 0.0, 1.0)


def normalize_for_show(image: np.ndarray) -> np.ndarray:
    """Scale any convolution output to 0..1 for display."""
    image = np.asarray(image, dtype=float)
    lo = float(image.min())
    hi = float(image.max())
    if hi == lo:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_cnn_and_mlp(
    seed: int = 0,
    train_size: int = 6000,
    test_size: int = 1000,
    epochs: int = 2,
    batch_size: int = 64,
    allow_download: bool = True,
    threads: int = 2,
) -> TorchVisionResult:
    """Train a tiny CNN and a plain MLP on the same image data.

    Fashion-MNIST is used when it is available. If it is not cached and cannot be
    downloaded, the function falls back to sklearn's 8x8 digits so the chapter still runs.
    """
    import random
    import time

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset, TensorDataset

    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    started = time.perf_counter()
    train_dataset, test_dataset, dataset_name, labels, image_size = _load_torchvision_or_digits(
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        allow_download=allow_download,
        Subset=Subset,
        TensorDataset=TensorDataset,
    )

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            reduced = image_size // 4
            self.classifier = nn.Linear(16 * reduced * reduced, len(labels))

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x.flatten(1))

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(image_size * image_size, 32),
                nn.ReLU(),
                nn.Linear(32, len(labels)),
            )

        def forward(self, x):
            return self.net(x)

    def train_one(model):
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        return model

    def evaluate(model):
        loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        predictions = []
        truths = []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb)
                predictions.append(out.argmax(dim=1).cpu().numpy())
                truths.append(yb.cpu().numpy())
        pred = np.concatenate(predictions)
        truth = np.concatenate(truths)
        return float((pred == truth).mean()), pred, truth

    torch.manual_seed(seed)
    mlp = train_one(TinyMLP())
    torch.manual_seed(seed)
    cnn = train_one(TinyCNN())

    mlp_accuracy, mlp_predictions, y_test = evaluate(mlp)
    cnn_accuracy, cnn_predictions, y_test = evaluate(cnn)

    test_images = _dataset_images(test_dataset, test_size)
    sample_tensor = test_dataset[0][0].unsqueeze(0)
    elapsed = time.perf_counter() - started

    return TorchVisionResult(
        dataset_name=dataset_name,
        labels=labels,
        image_size=image_size,
        cnn=cnn,
        mlp=mlp,
        cnn_accuracy=cnn_accuracy,
        mlp_accuracy=mlp_accuracy,
        cnn_params=parameter_count(cnn),
        mlp_params=parameter_count(mlp),
        cnn_predictions=cnn_predictions,
        mlp_predictions=mlp_predictions,
        y_test=y_test,
        test_images=test_images,
        sample_tensor=sample_tensor,
        elapsed=elapsed,
        epochs=epochs,
        train_size=len(train_dataset),
        test_size=len(test_dataset),
    )


def _load_torchvision_or_digits(train_size, test_size, seed, allow_download, Subset, TensorDataset):
    try:
        from torchvision import datasets, transforms

        transform = transforms.ToTensor()
        train_full = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=allow_download, transform=transform)
        test_full = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=allow_download, transform=transform)
        import torch

        train_count = min(train_size, len(train_full))
        test_count = min(test_size, len(test_full))
        generator = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_full), generator=generator)[:train_count].tolist()
        test_idx = torch.randperm(len(test_full), generator=generator)[:test_count].tolist()
        return Subset(train_full, train_idx), Subset(test_full, test_idx), "Fashion-MNIST", FASHION_LABELS, 28
    except Exception:
        return _digits_tensor_datasets(train_size, test_size, seed, TensorDataset)


def _digits_tensor_datasets(train_size, test_size, seed, TensorDataset):
    import torch

    X, y, images = load_digits()
    images = (images / 16.0).astype("float32")
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        images, y, images, test_size=0.25, random_state=seed, stratify=y
    )
    train_count = min(train_size, len(X_train))
    test_count = min(test_size, len(X_test))
    train_x = torch.tensor(X_train[:train_count]).unsqueeze(1)
    train_y = torch.tensor(y_train[:train_count], dtype=torch.long)
    test_x = torch.tensor(X_test[:test_count]).unsqueeze(1)
    test_y = torch.tensor(y_test[:test_count], dtype=torch.long)
    labels = tuple(str(i) for i in range(10))
    return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y), "sklearn digits fallback", labels, 8


def _dataset_images(dataset, limit: int) -> np.ndarray:
    images = []
    n = min(limit, len(dataset))
    for i in range(n):
        x, _ = dataset[i]
        images.append(x.squeeze(0).detach().cpu().numpy())
    return np.asarray(images)


def first_conv_filters(result: TorchVisionResult) -> np.ndarray:
    """The learned filters from the CNN's first sliding-window layer."""
    weights = result.cnn.features[0].weight.detach().cpu().numpy()
    return weights[:, 0, :, :]


def feature_maps(result: TorchVisionResult, limit: int = 8) -> np.ndarray:
    """Where the first learned filters light up on one test image."""
    import torch

    result.cnn.eval()
    with torch.no_grad():
        maps = result.cnn.features[:2](result.sample_tensor)
    return maps[0, :limit].detach().cpu().numpy()


def cnn_wrong_examples(result: TorchVisionResult, limit: int = 6):
    wrong = np.flatnonzero(result.cnn_predictions != result.y_test)
    rows = []
    for idx in wrong[:limit]:
        rows.append((result.test_images[idx], int(result.y_test[idx]), int(result.cnn_predictions[idx])))
    return rows


def model_comparison_table(result: TorchVisionResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "plain MLP", "test accuracy": result.mlp_accuracy, "numbers to learn": result.mlp_params},
            {"model": "tiny CNN", "test accuracy": result.cnn_accuracy, "numbers to learn": result.cnn_params},
        ]
    )


def plot_small_images(images, titles=None, cmap="gray_r", width=1.5, vcenter: bool = False):
    """A compact grid for filters, mistakes, and feature maps."""
    images = list(images)
    n = len(images)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, width * rows + 0.35))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        img = np.asarray(images[i], dtype=float)
        if vcenter:
            span = max(abs(float(img.min())), abs(float(img.max())), 1e-9)
            ax.imshow(img, cmap="coolwarm", vmin=-span, vmax=span, interpolation="nearest")
        else:
            show_image(img, ax=ax, cmap=cmap)
        if titles is not None:
            ax.set_title(str(titles[i]), fontsize=9)
    fig.tight_layout()
    return fig


def plot_confidences(probabilities: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(range(10), probabilities, color=ACCENT)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)
    ax.set_xlabel("digit")
    ax.set_ylabel("confidence")
    ax.set_title("What the digit model believes")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_kernel_demo(image: np.ndarray, kernel: np.ndarray):
    result = convolve2d_valid(image, kernel)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    show_image(image, ax=axes[0], title="image", cmap="gray_r")
    show_image(kernel, ax=axes[1], title="kernel", cmap="coolwarm", numbers=True)
    show_image(normalize_for_show(result), ax=axes[2], title="output", cmap="magma")
    fig.tight_layout()
    return fig, result
