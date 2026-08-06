"""Helpers for the unsupervised-learning chapters.

Part 5 is about what you can learn when nobody hands you the answers. The helpers here
keep the Streamlit pages and notebooks in agreement for kNN, k-means, and PCA.
"""

from __future__ import annotations

import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_sample_image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kidsml.datasets import cluster_blobs, digits, load_table, six_points_for_kmeans, toy_shape
from kidsml.plots import ACCENT, COOL, MUTED, PANEL, SHAPE, WARM, decision_boundary, scatter_2d, show_image


# ---------------------------------------------------------------------------
# Chapter 08: k nearest neighbours
# ---------------------------------------------------------------------------


def knn_hand_example():
    """Five labelled points and one query point, with pencil-friendly distances."""
    names = np.array(["A", "B", "C", "D", "E"])
    X = np.array([[3.0, 4.0], [6.0, 0.0], [0.0, 8.0], [9.0, 12.0], [16.0, 0.0]])
    labels = np.array(["red", "blue", "blue", "red", "red"])
    query = np.array([0.0, 0.0])
    return names, X, labels, query


def knn_distance_table():
    names, X, labels, query = knn_hand_example()
    rows = []
    for name, point, label in zip(names, X, labels):
        dx = float(point[0] - query[0])
        dy = float(point[1] - query[1])
        squared = dx * dx + dy * dy
        rows.append(
            {
                "point": name,
                "label": label,
                "x": int(point[0]),
                "y": int(point[1]),
                "dx² + dy²": int(squared),
                "distance": int(np.sqrt(squared)),
            }
        )
    return pd.DataFrame(rows).sort_values("dx² + dy²", ignore_index=True)


def knn_vote_table(k: int = 3):
    table = knn_distance_table().head(k)
    red = int((table["label"] == "red").sum())
    blue = int((table["label"] == "blue").sum())
    winner = "red" if red > blue else "blue" if blue > red else "tie"
    votes = pd.DataFrame({"label": ["red", "blue"], "votes": [red, blue]})
    return table, votes, winner


def plot_knn_hand(k: int = 3):
    names, X, labels, query = knn_hand_example()
    table, votes, winner = knn_vote_table(k)
    nearest_names = set(table["point"])
    kth = float(table["distance"].iloc[-1])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colours = np.where(labels == "red", WARM, COOL)
    ax.scatter(X[:, 0], X[:, 1], s=120, c=colours, edgecolors=PANEL, linewidths=1.2, zorder=3)
    ax.scatter([query[0]], [query[1]], marker="*", s=260, c=ACCENT, edgecolors="black", linewidths=0.8, zorder=4)
    for name, point in zip(names, X):
        ax.text(point[0] + 0.25, point[1] + 0.25, name, weight="bold")
        if name in nearest_names:
            ax.plot([query[0], point[0]], [query[1], point[1]], color=ACCENT, linewidth=2.0, alpha=0.8)
    circle = plt.Circle(query, kth, fill=False, color=ACCENT, linestyle="--", linewidth=2.2)
    ax.add_patch(circle)
    ax.set_xlim(-2, 18)
    ax.set_ylim(-2, 14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"k = {k}: {winner} wins ({int(votes.votes.max())} votes)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def plot_knn_hand_boundary(k: int = 1):
    _, X, labels, query = knn_hand_example()
    y = (labels == "red").astype(int)
    model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    decision_boundary(model.predict, X, y, ax=ax, steps=160, shade_confidence=False)
    ax.scatter([query[0]], [query[1]], marker="*", s=240, c=ACCENT, edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_title(f"The whole plane copies its {k} nearest neighbour(s)")
    return fig


def knn_toy_data(n: int = 170, noise: float = 0.24, seed: int = 4):
    return toy_shape("moons", n=n, noise=noise, seed=seed)


def plot_knn_play(k: int = 7, qx: float = 0.0, qy: float = 0.0, n: int = 170, noise: float = 0.24, seed: int = 4):
    X, y = knn_toy_data(n=n, noise=noise, seed=seed)
    model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    query = np.array([[qx, qy]])
    distances = np.sqrt(((X - query) ** 2).sum(axis=1))
    nearest = np.argsort(distances)[:k]
    pred = int(model.predict(query)[0])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    decision_boundary(model.predict, X, y, ax=ax, steps=120, shade_confidence=False)
    for i in nearest:
        ax.plot([qx, X[i, 0]], [qy, X[i, 1]], color=ACCENT, linewidth=1.5, alpha=0.75, zorder=4)
    ax.scatter(X[nearest, 0], X[nearest, 1], s=160, facecolors="none", edgecolors="black", linewidths=1.8, zorder=5)
    ax.scatter([qx], [qy], marker="*", s=300, c=ACCENT, edgecolors="black", linewidths=0.8, zorder=6)
    ax.set_title(f"The query copies class {pred}")
    blue = int((y[nearest] == 0).sum())
    red = int((y[nearest] == 1).sum())
    votes = pd.DataFrame({"neighbour class": ["blue", "red"], "votes": [blue, red]})
    return fig, votes


def knn_accuracy_curve(shape: str = "moons", n: int = 260, noise: float = 0.28, seed: int = 7):
    X, y = toy_shape(shape, n=n, noise=noise, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    rows = []
    for k in range(1, 42, 2):
        model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        rows.append({"k": k, "test accuracy": model.score(X_test, y_test)})
    return pd.DataFrame(rows)


def knn_timing_table(seed: int = 0):
    """Measure fit and predict time, using fresh random data each call."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in [200, 1000, 3000]:
        X = rng.normal(size=(n, 24))
        y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(scale=0.7, size=n) > 0).astype(int)
        Xq = rng.normal(size=(500, 24))
        for name, model in [
            ("kNN", KNeighborsClassifier(n_neighbors=7)),
            ("logistic regression", LogisticRegression(max_iter=300)),
        ]:
            start = time.perf_counter()
            model.fit(X, y)
            fit_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            model.predict(Xq)
            predict_ms = (time.perf_counter() - start) * 1000
            rows.append({"rows remembered": n, "model": name, "fit ms": fit_ms, "predict 500 ms": predict_ms})
    return pd.DataFrame(rows)


def penguin_knn_scores(k: int = 7):
    df = load_table("penguins").dropna(subset=["species", "beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"])
    X = df[["beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"]].to_numpy()
    y, _ = pd.factorize(df["species"])
    raw = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X, y, cv=5).mean()
    return pd.DataFrame(
        {
            "version": ["raw measurements", "scaled first"],
            "accuracy": [raw, scaled],
        }
    )


def digits_knn_score(k: int = 3, seed: int = 0):
    X, y, _ = digits()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    return float(model.score(X_test, y_test))


# ---------------------------------------------------------------------------
# Chapter 20: k-means
# ---------------------------------------------------------------------------


def kmeans_hand_points():
    X = six_points_for_kmeans()
    centres = np.array([[0.0, 0.0], [10.0, 10.0]])
    return X, centres


def assign_to_centres(X, centres):
    distances = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1), distances


def move_centres(X, labels, k: int):
    centres = []
    for i in range(k):
        members = X[labels == i]
        if len(members) == 0:
            centres.append(np.array([np.nan, np.nan]))
        else:
            centres.append(members.mean(axis=0))
    return np.asarray(centres)


def kmeans_hand_round():
    X, centres = kmeans_hand_points()
    labels, distances = assign_to_centres(X, centres)
    new_centres = move_centres(X, labels, 2)
    rows = []
    for i, point in enumerate(X):
        rows.append(
            {
                "point": f"P{i + 1}",
                "x": point[0],
                "y": point[1],
                "dist² to left centre": distances[i, 0],
                "dist² to right centre": distances[i, 1],
                "joins": "left" if labels[i] == 0 else "right",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(new_centres, columns=["new x", "new y"])


def kmeans_history(k: int = 3, seed: int = 0, bad_start: bool = False, n: int = 180):
    X, _ = cluster_blobs(n=n, k=3, spread=0.65, seed=2)
    rng = np.random.default_rng(seed)
    if bad_start and k == 3:
        centres = np.array([[-1.8, -9.8], [-1.0, -9.0], [-0.2, -2.2]])
    else:
        pick = rng.choice(len(X), size=k, replace=False)
        centres = X[pick].copy()
    stages = []
    labels = np.zeros(len(X), dtype=int)
    for round_id in range(6):
        labels, _ = assign_to_centres(X, centres)
        stages.append({"caption": f"assign: every dot joins the nearest centre (round {round_id + 1})", "X": X, "centres": centres.copy(), "labels": labels.copy()})
        new_centres = move_centres(X, labels, k)
        stages.append({"caption": f"move: each centre walks to the middle of its members (round {round_id + 1})", "X": X, "centres": new_centres.copy(), "labels": labels.copy()})
        if np.allclose(new_centres, centres):
            break
        centres = new_centres
    return stages


def plot_kmeans_stage(stage):
    X = stage["X"]
    labels = stage["labels"]
    centres = stage["centres"]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cmap = plt.get_cmap("tab10")
    for i in range(len(centres)):
        members = X[labels == i]
        if len(members):
            ax.scatter(members[:, 0], members[:, 1], s=42, color=cmap(i), edgecolors=PANEL, linewidths=0.6, alpha=0.9)
            for point in members:
                ax.plot([point[0], centres[i, 0]], [point[1], centres[i, 1]], color=MUTED, linewidth=0.6, alpha=0.35)
    ax.scatter(centres[:, 0], centres[:, 1], marker="X", s=280, c=PANEL, edgecolors=SHAPE, linewidths=1.6, zorder=4)
    ax.set_title(stage["caption"])
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return fig


def kmeans_elbow_data(kind: str = "obvious", max_k: int = 8):
    if kind == "obvious":
        X, _ = cluster_blobs(n=220, k=3, spread=0.55, seed=5)
    else:
        X, _ = cluster_blobs(n=220, k=4, spread=1.45, seed=8)
    rows = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        rows.append({"k": k, "inertia": model.inertia_})
    return X, pd.DataFrame(rows)


def plot_elbow(kind: str = "obvious"):
    _, table = kmeans_elbow_data(kind=kind)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(table["k"], table["inertia"], marker="o", color=ACCENT)
    ax.set_xlabel("number of centres, k")
    ax.set_ylabel("total distance to own centre")
    ax.set_title("An elbow plot" if kind == "obvious" else "Sometimes the elbow is mushy")
    return fig


def default_flower_image(max_side: int = 240):
    image = load_sample_image("flower.jpg")
    return resize_image_array(image, max_side=max_side)


def resize_image_array(image, max_side: int = 240):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        pil = Image.fromarray(arr)
        pil = pil.resize((int(w * scale), int(h * scale)))
        arr = np.asarray(pil)
    return arr[:, :, :3]


def uploaded_image_to_array(uploaded, max_side: int = 240):
    data = uploaded.read()
    image = Image.open(BytesIO(data)).convert("RGB")
    return resize_image_array(np.asarray(image), max_side=max_side)


def quantize_image(image, k: int = 5, sample_pixels: int = 5000, seed: int = 0):
    arr = resize_image_array(image)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3).astype(float) / 255.0
    rng = np.random.default_rng(seed)
    take = min(sample_pixels, len(pixels))
    sample = pixels[rng.choice(len(pixels), size=take, replace=False)]
    model = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(sample)
    labels = model.predict(pixels)
    palette = np.clip(model.cluster_centers_, 0, 1)
    rebuilt = palette[labels].reshape(h, w, 3)
    return rebuilt, palette


def plot_palette(palette):
    fig, ax = plt.subplots(figsize=(5.6, 1.1))
    swatches = palette.reshape(1, len(palette), 3)
    ax.imshow(swatches, interpolation="nearest")
    ax.set_xticks(range(len(palette)), [str(i + 1) for i in range(len(palette))])
    ax.set_yticks([])
    ax.set_title("The colours k-means found")
    ax.grid(False)
    return fig


def plot_kmeans_failure(shape: str = "moons"):
    X, y = toy_shape(shape, n=220, noise=0.12, seed=2)
    model = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    scatter_2d(X, model.labels_, ax=ax, labels=False)
    ax.set_title(f"k-means slices {shape} with round-ish chunks")
    return fig


def plot_dbscan_moons():
    X, y = toy_shape("moons", n=220, noise=0.12, seed=2)
    labels = DBSCAN(eps=0.28, min_samples=5).fit_predict(X)
    clean = np.where(labels < 0, 0, labels)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    scatter_2d(X, clean, ax=ax, labels=False)
    ax.set_title("DBSCAN follows the crescent shapes")
    return fig


def penguin_kmeans_table():
    df = load_table("penguins").dropna(subset=["species", "beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"])
    X = df[["beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"]].to_numpy()
    scaled = StandardScaler().fit_transform(X)
    clusters = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(scaled)
    table = pd.crosstab(pd.Series(clusters, name="cluster number"), df["species"])
    y, _ = pd.factorize(df["species"])
    score = adjusted_rand_score(y, clusters)
    return table, float(score)


# ---------------------------------------------------------------------------
# Chapter 21: PCA
# ---------------------------------------------------------------------------


def pca_hand_points():
    return np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 3.0], [4.0, 3.0]])


def spread(values):
    values = np.asarray(values, dtype=float)
    middle = values.mean()
    return float(((values - middle) ** 2).sum())


def pca_hand_table():
    X = pca_hand_points()
    return pd.DataFrame(
        {
            "point": ["A", "B", "C", "D"],
            "x shadow": X[:, 0],
            "y shadow": X[:, 1],
            "x - x middle": X[:, 0] - X[:, 0].mean(),
            "y - y middle": X[:, 1] - X[:, 1].mean(),
        }
    )


def make_shadow_cloud(n: int = 360, seed: int = 1):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 3)) * np.array([3.0, 1.0, 0.22])
    a = np.deg2rad(35)
    b = np.deg2rad(-25)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return raw @ (rz @ ry).T


def projection_basis(yaw_deg: float = 0.0, pitch_deg: float = 0.0):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    v = np.array([-np.sin(yaw) * np.sin(pitch), np.cos(yaw) * np.sin(pitch), np.cos(pitch)])
    return np.c_[u, v]


def shadow_projection(X, yaw_deg: float = 0.0, pitch_deg: float = 0.0):
    basis = projection_basis(yaw_deg, pitch_deg)
    return X @ basis


def variance_captured(X, shadow):
    total = float(np.var(X, axis=0).sum())
    kept = float(np.var(shadow, axis=0).sum())
    return kept / total


def pca_shadow_answer(X):
    model = PCA(n_components=2).fit(X)
    shadow = model.transform(X)
    return shadow, float(model.explained_variance_ratio_.sum())


def plot_shadow_3d(X):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker={"size": 3, "color": ACCENT, "opacity": 0.75},
            )
        ]
    )
    fig.update_layout(height=430, margin={"l": 0, "r": 0, "t": 25, "b": 0}, title="The object you can rotate")
    return fig


def plot_shadow_2d(shadow, title: str = "Your shadow"):
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.scatter(shadow[:, 0], shadow[:, 1], s=18, c=ACCENT, edgecolors=PANEL, linewidths=0.3, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("shadow left-right")
    ax.set_ylabel("shadow up-down")
    return fig


def digits_pca_embedding(n_components: int = 2):
    X, y, images = digits()
    model = PCA(n_components=n_components, random_state=0).fit(X)
    Z = model.transform(X)
    return X, y, images, model, Z


def plot_digits_pca():
    X, y, _, model, Z = digits_pca_embedding(2)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=16, alpha=0.75, edgecolors="none")
    ax.set_title("Digits squished to two PCA shadows")
    ax.set_xlabel("first shadow")
    ax.set_ylabel("second shadow")
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="true digit")
    return fig, float(model.explained_variance_ratio_.sum())


def reconstruct_digit(index: int = 0, n_components: int = 12):
    X, y, images = digits()
    full = PCA(n_components=64, random_state=0).fit(X)
    coeff = full.transform(X[index : index + 1]).copy()
    coeff[:, n_components:] = 0.0
    rebuilt = full.inverse_transform(coeff).reshape(8, 8)
    original = images[index]
    diff = np.abs(original - rebuilt)
    curve = np.cumsum(full.explained_variance_ratio_)
    return original, rebuilt, diff, curve, full, int(y[index])


def plot_reconstruction(index: int = 0, n_components: int = 12):
    original, rebuilt, diff, curve, _, digit = reconstruct_digit(index, n_components)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    show_image(original, ax=axes[0], title=f"original {digit}")
    show_image(rebuilt, ax=axes[1], title=f"rebuilt with {n_components}")
    show_image(diff, ax=axes[2], title="difference", cmap="magma")
    fig.tight_layout()
    return fig, curve


def plot_variance_curve(curve, n_components: int = 12):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs = np.arange(1, len(curve) + 1)
    ax.plot(xs, curve, color=ACCENT)
    ax.axvline(n_components, color=WARM, linestyle="--", linewidth=1.8)
    ax.set_xlabel("components kept")
    ax.set_ylabel("information kept")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative explained variance")
    return fig


def plot_eigendigits(count: int = 8):
    _, _, _, _, full, _ = reconstruct_digit(0, 12)
    fig, axes = plt.subplots(1, count, figsize=(count * 1.1, 1.45))
    for i, ax in enumerate(np.ravel(axes)):
        show_image(full.components_[i].reshape(8, 8), ax=ax, title=str(i + 1), cmap="gray")
    fig.suptitle("Principal components: ghost digits", y=1.05)
    return fig


def digits_tsne_embedding(n: int = 600, seed: int = 0):
    X, y, _ = digits()
    X = X[:n]
    y = y[:n]
    X30 = PCA(n_components=30, random_state=seed).fit_transform(X)
    model = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=seed, max_iter=500)
    Z = model.fit_transform(X30)
    return Z, y


def plot_digits_tsne(n: int = 600, seed: int = 0):
    Z, y = digits_tsne_embedding(n=n, seed=seed)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=18, alpha=0.78, edgecolors="none")
    ax.set_title("t-SNE keeps neighbours together")
    ax.set_xlabel("t-SNE direction 1")
    ax.set_ylabel("t-SNE direction 2")
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="true digit")
    return fig


def plot_pca_linear_failure():
    X2, y = toy_shape("circles", n=260, noise=0.06, seed=3)
    r = np.sqrt((X2 ** 2).sum(axis=1))
    X3 = np.c_[X2, r]
    Z = PCA(n_components=2, random_state=0).fit_transform(X3)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    scatter_2d(Z, y, ax=ax)
    ax.set_title("A flat PCA shadow still tangles the lifted circles")
    ax.set_xlabel("PCA shadow 1")
    ax.set_ylabel("PCA shadow 2")
    return fig


def penguin_pca_table():
    df = load_table("penguins").dropna(subset=["species", "beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"])
    features = ["beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"]
    X = StandardScaler().fit_transform(df[features].to_numpy())
    model = PCA(n_components=2, random_state=0).fit(X)
    loadings = pd.DataFrame({"feature": features, "first shadow weight": model.components_[0]})
    return loadings, float(model.explained_variance_ratio_[0])
