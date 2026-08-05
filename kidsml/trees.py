"""Tree, ensemble, SVM, and evaluation helpers for Part 1.

The chapter pages and notebooks share the same small calculations and pictures from
here, so the app and notebook stories stay in sync.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

from kidsml.datasets import MONSTER_SECRET_RULE, load_table, toy_shape, wiggle
from kidsml.plots import ACCENT, COOL, MUTED, WARM, decision_boundary, scatter_2d


# ---------------------------------------------------------------------------
# Chapter 05: decision trees
# ---------------------------------------------------------------------------


def gini(yes_count: int, no_count: int) -> float:
    """How mixed a two-class bucket is. Zero means one kind only."""
    total = yes_count + no_count
    if total == 0:
        return 0.0
    p_yes = yes_count / total
    p_no = no_count / total
    return float(1.0 - p_yes * p_yes - p_no * p_no)


def creature_feature_names() -> list[str]:
    return ["has_wings", "bigger_than_cat", "has_feathers", "lives_in_water"]


def creature_split_table() -> pd.DataFrame:
    """Try every first question in the tiny creatures table."""
    df = load_table("creatures")
    rows = []
    for feature in creature_feature_names():
        yes_bucket = df[df[feature] == "yes"]
        no_bucket = df[df[feature] == "no"]
        yes_fly = int((yes_bucket["can_fly"] == "yes").sum())
        yes_no_fly = int((yes_bucket["can_fly"] == "no").sum())
        no_fly = int((no_bucket["can_fly"] == "yes").sum())
        no_no_fly = int((no_bucket["can_fly"] == "no").sum())
        yes_gini = gini(yes_fly, yes_no_fly)
        no_gini = gini(no_fly, no_no_fly)
        weighted = (len(yes_bucket) * yes_gini + len(no_bucket) * no_gini) / len(df)
        rows.append(
            {
                "first question": feature,
                "yes bucket": f"{yes_fly} fly, {yes_no_fly} do not",
                "yes gini": round(yes_gini, 3),
                "no bucket": f"{no_fly} fly, {no_no_fly} do not",
                "no gini": round(no_gini, 3),
                "weighted mix": round(weighted, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted mix", ignore_index=True)


def fit_creature_tree(max_depth: int = 3) -> tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series]:
    df = load_table("creatures")
    X = df[creature_feature_names()].replace({"no": 0, "yes": 1})
    y = (df["can_fly"] == "yes").astype(int)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    return model, X, y


def tree_depth_scores(shape: str = "moons", n: int = 220, noise: float = 0.2, seed: int = 1) -> pd.DataFrame:
    X, y = toy_shape(shape, n=n, noise=noise, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    rows = []
    for depth in [1, 2, 3, 4, 6, 8, 12, 20]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
        rows.append(
            {
                "max_depth": depth,
                "train accuracy": model.score(X_train, y_train),
                "test accuracy": model.score(X_test, y_test),
            }
        )
    return pd.DataFrame(rows)


def fit_tree_shape(shape: str, max_depth: int, n: int = 220, noise: float = 0.2, seed: int = 1):
    X, y = toy_shape(shape, n=n, noise=noise, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def plot_decision_tree(model: DecisionTreeClassifier, feature_names: list[str], class_names: list[str], ax=None):
    ax = ax or plt.gca()
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        ax=ax,
        fontsize=9,
    )
    return ax


def mushroom_tree(max_depth: int = 4, seed: int = 2):
    df = load_table("mushrooms")
    X = pd.get_dummies(df.drop(columns=["edible"]))
    y = (df["edible"] == "edible").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X_train, y_train)
    scores = {
        "train": model.score(X_train, y_train),
        "test": model.score(X_test, y_test),
        "top_question": X.columns[int(model.tree_.feature[0])],
    }
    text = export_text(model, feature_names=list(X.columns), max_depth=max_depth)
    return model, X_train, X_test, y_train, y_test, scores, text


def shallow_mushroom_scores(max_depth: int = 8) -> pd.DataFrame:
    rows = []
    for depth in range(1, max_depth + 1):
        _, _, _, _, _, scores, _ = mushroom_tree(max_depth=depth)
        rows.append({"depth": depth, "train accuracy": scores["train"], "test accuracy": scores["test"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chapter 06: ensembles and boosting
# ---------------------------------------------------------------------------


def tiny_vote_table() -> pd.DataFrame:
    data = {
        "test point": ["A", "B", "C", "D"],
        "tree 1": ["red", "blue", "red", "blue"],
        "tree 2": ["red", "blue", "blue", "blue"],
        "tree 3": ["blue", "blue", "red", "red"],
        "tree 4": ["red", "blue", "red", "blue"],
        "tree 5": ["red", "red", "red", "blue"],
    }
    table = pd.DataFrame(data)
    winners = []
    for _, row in table.iterrows():
        votes = [row[f"tree {i}"] for i in range(1, 6)]
        winners.append(max(set(votes), key=votes.count))
    table["crowd vote"] = winners
    return table


def fit_tree_and_forest(shape: str = "moons", n_estimators: int = 30, n: int = 240, noise: float = 0.25, seed: int = 4):
    X, y = toy_shape(shape, n=n, noise=noise, seed=seed)
    tree = DecisionTreeClassifier(max_depth=8, random_state=seed).fit(X, y)
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=8, random_state=seed).fit(X, y)
    return X, y, tree, forest


def forest_vote_counts(forest: RandomForestClassifier, point: np.ndarray) -> dict[str, int]:
    point = np.asarray(point, dtype=float).reshape(1, -1)
    votes = [int(tree.predict(point)[0]) for tree in forest.estimators_]
    return {"blue": votes.count(0), "red": votes.count(1)}


def boosting_trace(n_steps: int = 20, learning_rate: float = 0.25, max_depth: int = 1, n: int = 50, noise: float = 0.22, seed: int = 0):
    x, y = wiggle(n=n, noise=noise, seed=seed)
    X = x.reshape(-1, 1)
    x_grid = np.linspace(x.min(), x.max(), 240)
    grid = x_grid.reshape(-1, 1)
    running = np.zeros_like(y, dtype=float)
    running_grid = np.zeros_like(x_grid, dtype=float)
    stages = []
    for step in range(1, n_steps + 1):
        residual = y - running
        stump = DecisionTreeRegressor(max_depth=max_depth, random_state=step).fit(X, residual)
        newest = learning_rate * stump.predict(X)
        newest_grid = learning_rate * stump.predict(grid)
        running = running + newest
        running_grid = running_grid + newest_grid
        stages.append(
            {
                "step": step,
                "running": running.copy(),
                "running_grid": running_grid.copy(),
                "residual": (y - running).copy(),
                "newest": newest.copy(),
                "newest_grid": newest_grid.copy(),
            }
        )
    return {"x": x, "y": y, "x_grid": x_grid, "stages": stages}


def tiny_boosting_table() -> pd.DataFrame:
    actual = np.array([2.0, 4.0, 8.0, 10.0])
    first = np.full(4, 5.0)
    residual1 = actual - first
    second = 0.5 * residual1
    after_second = first + second
    residual2 = actual - after_second
    return pd.DataFrame(
        {
            "actual": actual,
            "first guess": first,
            "leftover": residual1,
            "fix half": second,
            "new guess": after_second,
            "new leftover": residual2,
        }
    )


def monster_xy():
    df = load_table("monsters")
    X_words = df.drop(columns=["name", "is_boss"])
    X = pd.get_dummies(X_words)
    y = (df["is_boss"] == "yes").astype(int)
    return X, y


def grouped_importances(feature_names, importances) -> pd.DataFrame:
    groups = {"attack": 0.0, "magic": 0.0, "speed": 0.0, "defense": 0.0, "height_cm": 0.0, "weight_kg": 0.0, "element": 0.0, "home": 0.0}
    for name, value in zip(feature_names, importances):
        group = name.split("_")[0]
        if name in groups:
            group = name
        if group in groups:
            groups[group] += float(value)
    out = pd.DataFrame({"feature group": list(groups), "importance": list(groups.values())})
    return out.sort_values("importance", ascending=False, ignore_index=True)


def monster_models(seed: int = 3):
    X, y = monster_xy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    forest = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=seed).fit(X_train, y_train)
    boost = GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=2, random_state=seed).fit(X_train, y_train)
    rows = []
    for name, model in [("random forest", forest), ("gradient boosting", boost)]:
        rows.append({"model": name, "train accuracy": model.score(X_train, y_train), "test accuracy": model.score(X_test, y_test)})
    imp = grouped_importances(X.columns, forest.feature_importances_)
    return pd.DataFrame(rows), imp, MONSTER_SECRET_RULE


# ---------------------------------------------------------------------------
# Chapter 07: SVMs
# ---------------------------------------------------------------------------


def svm_hand_points():
    X = np.array([[1.0, 1.0], [1.0, 3.0], [2.0, 2.0], [5.0, 1.0], [5.0, 3.0], [4.0, 2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    candidates = pd.DataFrame(
        {
            "candidate road": ["x = 2.5", "x = 3.0"],
            "gap to nearest blue": [0.5, 1.0],
            "gap to nearest red": [1.5, 1.0],
            "road width": [2.0, 2.0],
            "smallest safety gap": [0.5, 1.0],
        }
    )
    return X, y, candidates


def svm_demo_points():
    X = np.array(
        [
            [-2.2, -1.0], [-2.0, 0.9], [-1.4, -0.2], [-1.0, 1.7],
            [1.0, -1.4], [1.5, 0.2], [2.1, -0.7], [2.4, 1.3],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def fit_linear_svm(remove: str = "none", C: float = 100.0):
    X, y = svm_demo_points()
    keep = np.ones(len(y), dtype=bool)
    if remove == "non-support":
        keep[0] = False
    elif remove == "support":
        keep[3] = False
    X2, y2 = X[keep], y[keep]
    model = SVC(kernel="linear", C=C).fit(X2, y2)
    return model, X2, y2


def plot_linear_svm_margin(model: SVC, X, y, ax=None, title: str | None = None):
    ax = ax or plt.gca()
    scatter_2d(X, y, ax=ax)
    w = model.coef_[0]
    b = model.intercept_[0]
    x_lo, x_hi = ax.get_xlim()
    xs = np.linspace(x_lo, x_hi, 200)
    centre = -(w[0] * xs + b) / w[1]
    upper = -(w[0] * xs + b - 1) / w[1]
    lower = -(w[0] * xs + b + 1) / w[1]
    ax.fill_between(xs, lower, upper, color=ACCENT, alpha=0.16, label="the road")
    ax.plot(xs, centre, color=ACCENT, label="centre line")
    ax.plot(xs, upper, color=MUTED, linestyle="--", linewidth=1.2)
    ax.plot(xs, lower, color=MUTED, linestyle="--", linewidth=1.2)
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150, facecolors="none", edgecolors="black", linewidths=1.8, label="support vectors")
    ax.set_xlim(x_lo, x_hi)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def fit_svm_shape(shape: str, kernel: str = "rbf", C: float = 3.0, gamma: float = 1.0, degree: int = 3, n: int = 220, noise: float = 0.2, seed: int = 0):
    X, y = toy_shape(shape, n=n, noise=noise, seed=seed)
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma, degree=degree))
    model.fit(X, y)
    return X, y, model


def fit_circles_lifted(seed: int = 2, noise: float = 0.12):
    X, y = toy_shape("circles", n=220, noise=noise, seed=seed)
    r2 = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)
    lifted = np.c_[X, r2]
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=2.0)).fit(lifted, y)

    def predict(grid):
        grid_r2 = (grid[:, 0] ** 2 + grid[:, 1] ** 2).reshape(-1, 1)
        return model.predict(np.c_[grid, grid_r2])

    return X, y, predict


def penguin_svm():
    df = load_table("penguins").dropna(subset=["species", "beak_length_mm", "beak_depth_mm"])
    X = df[["beak_length_mm", "beak_depth_mm"]].to_numpy()
    y_codes, species = pd.factorize(df["species"])
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)).fit(X, y_codes)
    scaler = model.named_steps["standardscaler"]
    svc = model.named_steps["svc"]
    support_original = scaler.inverse_transform(svc.support_vectors_)
    return X, y_codes, list(species), model, support_original


# ---------------------------------------------------------------------------
# Chapter 08: model zoo and evaluation
# ---------------------------------------------------------------------------


def zoo_model_list():
    return [
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
        ("tree", DecisionTreeClassifier(max_depth=5, random_state=0)),
        ("forest", RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0)),
        ("boosting", GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.08, random_state=0)),
        ("linear SVM", make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))),
        ("rbf SVM", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma=1.0))),
        ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
    ]


MODEL_PERSONALITIES = {
    "logistic": "one straight line",
    "tree": "boxes and stairs",
    "forest": "many boxy votes",
    "boosting": "little fixes in a row",
    "linear SVM": "the widest straight road",
    "rbf SVM": "smooth islands",
    "kNN": "ask nearby points",
}


def fit_zoo(shape: str = "moons", n: int = 180, noise: float = 0.2, seed: int = 0):
    X, y = toy_shape(shape, n=n, noise=noise, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    entries = []
    for name, model in zoo_model_list():
        fitted = clone(model).fit(X_train, y_train)
        entries.append({"name": name, "model": fitted, "score": fitted.score(X_test, y_test)})
    return X, y, X_test, y_test, entries


def plot_zoo(shape: str = "moons", n: int = 180, noise: float = 0.2, seed: int = 0):
    X, y, _, _, entries = fit_zoo(shape=shape, n=n, noise=noise, seed=seed)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    flat = axes.ravel()
    for ax, entry in zip(flat, entries):
        decision_boundary(entry["model"].predict, X, y, ax=ax, steps=130, shade_confidence=False)
        ax.set_title(f"{entry['name']}  {entry['score']:.0%}\n{MODEL_PERSONALITIES[entry['name']]}")
    flat[-1].axis("off")
    fig.tight_layout()
    return fig


def deep_tree_train_test(shape: str = "moons", n: int = 220, noise: float = 0.28, seed: int = 4):
    X, y = toy_shape(shape, n=n, noise=noise, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    model = DecisionTreeClassifier(max_depth=20, random_state=0).fit(X_train, y_train)
    return {
        "train accuracy": model.score(X_train, y_train),
        "test accuracy": model.score(X_test, y_test),
        "wrong self-test": model.score(X_train, y_train),
    }


def split_bounce_scores(test_size: float = 0.3, max_seed: int = 10) -> pd.DataFrame:
    X, y = toy_shape("moons", n=180, noise=0.28, seed=0)
    rows = []
    for seed in range(max_seed + 1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        model = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X_train, y_train)
        rows.append({"seed": seed, "test accuracy": model.score(X_test, y_test)})
    return pd.DataFrame(rows)


def fold_scores(model=None, k: int = 5):
    X, y = toy_shape("moons", n=160, noise=0.25, seed=7)
    if model is None:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma=1.0))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv)
    return scores


def plot_folds(k: int = 5, ax=None):
    ax = ax or plt.gca()
    n = 10
    for fold in range(k):
        for row in range(k):
            color = WARM if row == fold else COOL
            ax.add_patch(Rectangle((row * 2, k - fold - 1), 2, 0.8, facecolor=color, alpha=0.75, edgecolor="white"))
    ax.set_xlim(0, n)
    ax.set_ylim(0, k)
    ax.set_xticks([1, 3, 5, 7, 9], ["part 1", "part 2", "part 3", "part 4", "part 5"], fontsize=8)
    ax.set_yticks([0.4, 1.4, 2.4, 3.4, 4.4], ["round 5", "round 4", "round 3", "round 2", "round 1"], fontsize=8)
    ax.set_title("Red is the held-out test fold")
    ax.grid(False)
    return ax


def lopsided_baseline(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 2))
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    model = DummyClassifier(strategy="most_frequent").fit(X, y)
    return accuracy_score(y, model.predict(X))


def penguin_leaderboard() -> pd.DataFrame:
    df = load_table("penguins").dropna(subset=["species", "beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"])
    X = df[["beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"]].to_numpy()
    y, _ = pd.factorize(df["species"])
    models = [
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("forest", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)),
        ("boosting", GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.08, random_state=0)),
        ("linear SVM", make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))),
        ("rbf SVM", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma=0.7))),
        ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
        ("baseline", DummyClassifier(strategy="most_frequent")),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    rows = []
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=cv)
        rows.append({"model": name, "mean": scores.mean(), "spread": scores.std(), "scores": ", ".join(f"{s:.2f}" for s in scores)})
    return pd.DataFrame(rows).sort_values("mean", ascending=False, ignore_index=True)
