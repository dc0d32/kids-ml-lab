"""Extra helpers for the picture chapters (18 and 19).

This lives in its own module so the two vision chapters can share a helper
without touching ``kidsml/vision.py`` while other chapters are being edited.

Right now it holds one thing: a cheap way to peek at a few real Fashion-MNIST
pictures *before* any training happens, so a reader can see what the data looks
like before being asked to predict which model wins on it.
"""

from __future__ import annotations

import numpy as np

from kidsml.vision import DATA_DIR, FASHION_LABELS

# A spread of classes, with the look-alike tops (shirt, pullover, coat, T-shirt)
# up front so the reader can see how easy they are to confuse, then some
# footwear so the strip is not all one shape.
_PREVIEW_CLASSES = (6, 2, 4, 0, 7, 5, 9, 1)


def fashion_preview(allow_download: bool = True):
    """A few real Fashion-MNIST pictures with their class names.

    Returns ``(images, names, dataset_name, image_size)`` where ``images`` is a
    list of 2D float arrays in 0..1 and ``names`` are the human class labels.

    Loading the test set does not train anything, so this is cheap. The first
    call downloads Fashion-MNIST if it is not cached yet (the same download the
    chapter needs anyway). If it is unavailable, this falls back to sklearn's
    8x8 digits so the chapter still shows something real.
    """
    try:
        from torchvision import datasets, transforms

        transform = transforms.ToTensor()
        test_full = datasets.FashionMNIST(
            root=str(DATA_DIR), train=False, download=allow_download, transform=transform
        )
        targets = np.asarray(test_full.targets)
        images = []
        names = []
        for label in _PREVIEW_CLASSES:
            idx = int(np.flatnonzero(targets == label)[0])
            image, _ = test_full[idx]
            images.append(image.squeeze(0).detach().cpu().numpy())
            names.append(FASHION_LABELS[label])
        return images, names, "Fashion-MNIST", 28
    except Exception:
        return _digits_preview()


def _digits_preview():
    from kidsml.datasets import digits

    _, y, images = digits()
    picked = []
    names = []
    for label in range(8):
        idx = int(np.flatnonzero(y == label)[0])
        picked.append(images[idx] / 16.0)
        names.append(str(label))
    return picked, names, "sklearn digits fallback", 8
