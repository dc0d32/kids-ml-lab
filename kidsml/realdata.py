"""Helpers for the messy-table chapters.

Part 2 leaves the flat two-column toy world. These helpers keep the page and notebook
stories in sync while the tables gain word columns, missing values, baselines, leakage,
and failure checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from kidsml import datasets

REAL_TABLES = ("penguins", "mushrooms", "monsters", "bikes")


# ---------------------------------------------------------------------------
# Chapter 10: real tables
# ---------------------------------------------------------------------------


def table_overview(name: str) -> dict:
    """Small facts a kid can use before any model appears."""
    df = datasets.load_table(name)
    target = datasets.target_of(name)
    missing = df.isna().sum()
    return {
        "name": name,
        "target": target,
        "rows": len(df),
        "columns": len(df.columns),
        "head": df.head(5),
        "dtypes": pd.DataFrame({"column": df.columns, "kind": [str(df[c].dtype) for c in df.columns]}),
        "missing": pd.DataFrame({"column": missing.index, "missing cells": missing.values}),
        "target_counts": df[target].value_counts(dropna=False).rename_axis(target).reset_index(name="rows"),
    }


def weather_one_hot_demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    before = pd.DataFrame(
        {
            "day": ["Mon", "Tue", "Wed"],
            "weather": ["clear", "rain", "misty"],
        }
    )
    after = before[["day"]].copy()
    for kind in ["clear", "misty", "rain", "storm"]:
        after[f"weather = {kind}"] = (before["weather"] == kind).astype(int)
    return before, after


def weather_hand_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": ["Mon", "Tue", "Wed", "Thu"],
            "weather": ["clear", "rain", "storm", "misty"],
        }
    )


def suggested_features(name: str) -> list[str]:
    target = datasets.target_of(name)
    skip = {target}
    if name == "monsters":
        skip.add("name")
    if name == "bikes":
        skip.add("date")
    return [c for c in datasets.load_table(name).columns if c not in skip]


def penguin_missing_rows() -> pd.DataFrame:
    df = datasets.load_table("penguins")
    return df[df.isna().any(axis=1)].reset_index(drop=True)


def fill_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].mean())
        else:
            common = out[col].mode(dropna=True)
            fill = "missing" if common.empty else common.iloc[0]
            out[col] = out[col].fillna(fill)
    return out


def one_hot_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    parts = []
    source = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric = df[[col]].astype(float)
            parts.append(numeric)
            source[col] = col
        else:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, prefix_sep=" = ", dtype=int)
            parts.append(dummies)
            for dummy_col in dummies.columns:
                source[dummy_col] = col
    if not parts:
        return pd.DataFrame(index=df.index), source
    return pd.concat(parts, axis=1), source


def _is_regression(name: str) -> bool:
    target = datasets.target_of(name)
    df = datasets.load_table(name)
    return pd.api.types.is_numeric_dtype(df[target])


def _prepared_table(
    name: str,
    features: list[str] | None = None,
    missing: str = "drop",
    max_rows: int | None = 1000,
    extra: dict[str, object] | None = None,
):
    df = datasets.load_table(name).copy()
    if extra:
        for col, values in extra.items():
            df[col] = values
    target = datasets.target_of(name)
    if features is None:
        features = suggested_features(name)
    work = df[features + [target]].copy()
    work = work.dropna(subset=[target])
    if missing == "drop":
        work = work.dropna()
    elif missing == "fill":
        work = fill_missing_values(work, features)
    else:
        raise ValueError("missing must be 'drop' or 'fill'")
    if max_rows is not None and len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=0)
    X_raw = work[features]
    X, source = one_hot_frame(X_raw)
    y = work[target]
    return work, X_raw, X, y, source


def train_table_model(
    name: str,
    features: list[str] | None = None,
    missing: str = "drop",
    max_rows: int | None = 1000,
    random_state: int = 0,
    extra: dict[str, object] | None = None,
) -> dict:
    work, X_raw, X, y, source = _prepared_table(name, features, missing, max_rows, extra)
    regression = _is_regression(name)
    stratify = None
    if not regression and y.value_counts().min() >= 2:
        stratify = y
    split = train_test_split(
        X, y, X_raw, work, test_size=0.30, random_state=random_state, stratify=stratify
    )
    X_train, X_test, y_train, y_test, raw_train, raw_test, rows_train, rows_test = split
    if regression:
        model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=random_state, n_jobs=1)
        baseline = DummyRegressor(strategy="mean")
        score_name = "R² score"
    else:
        model = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=random_state, n_jobs=1)
        baseline = DummyClassifier(strategy="most_frequent")
        score_name = "accuracy"
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    importances = feature_importance_table(model, X.columns, source)
    return {
        "name": name,
        "features": list(X_raw.columns),
        "target": datasets.target_of(name),
        "rows_used": len(work),
        "model": model,
        "baseline": baseline,
        "X_test": X_test,
        "y_test": y_test,
        "raw_test": raw_test,
        "rows_test": rows_test,
        "model_score": float(model.score(X_test, y_test)),
        "baseline_score": float(baseline.score(X_test, y_test)),
        "score_name": score_name,
        "importances": importances,
    }


def feature_importance_table(model, encoded_columns, source: dict[str, str]) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", np.zeros(len(encoded_columns)))
    rows = []
    for col, value in zip(encoded_columns, values):
        rows.append({"column": source.get(col, col), "importance": float(value)})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame({"column": [], "importance": []})
    return table.groupby("column", as_index=False)["importance"].sum().sort_values("importance", ascending=False)


def all_dataset_scores() -> pd.DataFrame:
    rows = []
    for name in REAL_TABLES:
        result = train_table_model(name)
        rows.append(
            {
                "dataset": name,
                "rows used": result["rows_used"],
                "kind of score": result["score_name"],
                "baseline": result["baseline_score"],
                "model": result["model_score"],
            }
        )
    return pd.DataFrame(rows)


def penguin_missing_scores() -> pd.DataFrame:
    rows = []
    for missing in ["drop", "fill"]:
        result = train_table_model("penguins", missing=missing, max_rows=None, random_state=2)
        rows.append(
            {
                "choice": "drop rows with blanks" if missing == "drop" else "fill blanks with averages/common words",
                "rows used": result["rows_used"],
                "baseline accuracy": result["baseline_score"],
                "model accuracy": result["model_score"],
            }
        )
    return pd.DataFrame(rows)


def penguin_confusion() -> dict:
    features = ["island", "beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g", "sex"]
    result = train_table_model("penguins", features=features, max_rows=None, random_state=4)
    labels = sorted(result["y_test"].unique())
    pred = result["model"].predict(result["X_test"])
    cm = confusion_matrix(result["y_test"], pred, labels=labels)
    numeric = ["beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g"]
    ranked = result["importances"][result["importances"]["column"].isin(numeric)]
    top = ranked["column"].head(2).tolist()
    if len(top) < 2:
        top = ["beak_length_mm", "beak_depth_mm"]
    clean = datasets.load_table("penguins").dropna().reset_index(drop=True)
    return {"cm": cm, "labels": labels, "top": top, "data": clean, "score": result["model_score"]}


def bike_regression() -> dict:
    features = [
        "season", "month", "weekday", "is_holiday", "is_workday", "weather",
        "temp_c", "feels_like_c", "humidity_pct", "wind_kmh",
    ]
    result = train_table_model("bikes", features=features, max_rows=None, random_state=5)
    pred = result["model"].predict(result["X_test"])
    full = datasets.load_table("bikes")
    rows = full.loc[result["rows_test"].index, ["date", "season", "weather", "temp_c", "humidity_pct", "rentals"]].copy()
    rows["predicted"] = np.round(pred).astype(int)
    rows["mistake"] = rows["predicted"] - rows["rentals"]
    rows["absolute mistake"] = rows["mistake"].abs()
    worst = rows.sort_values("absolute mistake", ascending=False).head(8)
    return {"result": result, "predicted": pred, "rows": rows, "worst": worst}


# ---------------------------------------------------------------------------
# Chapter 11: failure modes
# ---------------------------------------------------------------------------


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def rare_disease_scores(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    sick = rng.choice(n, size=max(1, n // 100), replace=False)
    y[sick] = 1
    score = rng.beta(1.2, 9.0, size=n) * 0.85
    score[y == 1] = 0.12 + rng.beta(3.0, 2.0, size=int(y.sum())) * 0.82
    return pd.DataFrame({"really sick?": y, "model worry score": score})


def threshold_report(threshold: float, n: int = 1000, seed: int = 0) -> dict:
    df = rare_disease_scores(n=n, seed=seed)
    y = df["really sick?"].to_numpy()
    pred = (df["model worry score"].to_numpy() >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    metrics = metrics_from_counts(tp, fp, fn, tn)
    cm = np.array([[tp, fn], [fp, tn]])
    return {"table": df, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "metrics": metrics, "cm": cm}


def always_healthy_accuracy(n: int = 1000) -> float:
    df = rare_disease_scores(n=n)
    return float((df["really sick?"] == 0).mean())


def leakage_scores() -> pd.DataFrame:
    df = datasets.load_table("monsters")
    features = [c for c in df.columns if c not in ["is_boss", "name"]]
    base = train_table_model("monsters", features=features, random_state=7)
    secret_copy = np.where(df["is_boss"] == "yes", "ruby", "pebble")
    leaked = train_table_model(
        "monsters", features=features + ["badge_color"], random_state=7, extra={"badge_color": secret_copy}
    )
    bike = datasets.load_table("bikes")
    bike_features = ["season", "month", "weather", "temp_c", "humidity_pct", "wind_kmh"]
    bike_base = train_table_model("bikes", features=bike_features, max_rows=None, random_state=8)
    rental_receipt = bike["rentals"]
    bike_leak = _bike_leakage_model(bike_features + ["rental_receipt"], rental_receipt)
    return pd.DataFrame(
        [
            {"demo": "monsters, honest columns", "score": base["model_score"], "what happened": "normal"},
            {"demo": "monsters, with badge_color", "score": leaked["model_score"], "what happened": "the answer was copied"},
            {"demo": "bikes, weather columns", "score": bike_base["model_score"], "what happened": "normal"},
            {"demo": "bikes, with rental_receipt", "score": bike_leak, "what happened": "a receipt leaked the answer"},
        ]
    )


def _bike_leakage_model(features: list[str], rental_receipt) -> float:
    work, _, X, y, _ = _prepared_table(
        "bikes", features=features, missing="drop", max_rows=None, extra={"rental_receipt": rental_receipt}
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=8)
    model = DecisionTreeRegressor(random_state=8)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def bias_toy_data(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = np.where(np.arange(n) % 2 == 0, "green", "purple")
    skill = rng.integers(40, 101, size=n)
    portfolio = np.clip(skill + rng.normal(0, 8, size=n), 0, 100).round().astype(int)
    qualified = skill >= 70
    hired_before = np.where(group == "green", skill >= 70, skill >= 85)
    return pd.DataFrame(
        {
            "group": group,
            "skill_score": skill,
            "portfolio_score": portfolio,
            "qualified?": np.where(qualified, "yes", "no"),
            "historical_hired?": np.where(hired_before, "yes", "no"),
        }
    )


def bias_report() -> dict:
    df = bias_toy_data()
    X_raw = df[["group", "skill_score", "portfolio_score"]]
    X, _ = one_hot_frame(X_raw)
    y = df["historical_hired?"]
    X_train, X_test, y_train, y_test, rows_train, rows_test = train_test_split(
        X, y, df, test_size=0.35, random_state=3, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=3, random_state=3)
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=rows_test.index)
    rows = rows_test.copy()
    rows["model says hire?"] = pred
    rows["right about history?"] = rows["model says hire?"] == rows["historical_hired?"]
    rows["qualified and accepted?"] = (rows["qualified?"] == "yes") & (rows["model says hire?"] == "yes")
    summary_rows = []
    for group_name, part in rows.groupby("group"):
        qualified = part[part["qualified?"] == "yes"]
        summary_rows.append(
            {
                "group": group_name,
                "accuracy against old labels": float(part["right about history?"].mean()),
                "qualified people accepted": float(qualified["qualified and accepted?"].mean()) if len(qualified) else 0.0,
                "rows checked": len(part),
            }
        )
    examples = pd.DataFrame(
        {
            "group": ["green", "purple"],
            "skill_score": [75, 75],
            "portfolio_score": [75, 75],
        }
    )
    examples_X, _ = one_hot_frame(examples)
    examples_X = examples_X.reindex(columns=X.columns, fill_value=0)
    examples["model says hire?"] = model.predict(examples_X)
    return {
        "data": df.head(12),
        "overall": float(model.score(X_test, y_test)),
        "summary": pd.DataFrame(summary_rows),
        "examples": examples,
    }


def moons_out_of_world(span: float = 8.0, seed: int = 2) -> dict:
    from kidsml.datasets import toy_shape

    X, y = toy_shape("moons", n=260, noise=0.18, seed=seed)
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(X, y)
    xs = np.linspace(-span, span, 180)
    ys = np.linspace(-span, span, 180)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid)
    confidence = proba.max(axis=1).reshape(xx.shape)
    far = np.array([[span * 0.9, span * 0.9]])
    far_proba = model.predict_proba(far)[0]
    far_guess = int(model.predict(far)[0])
    return {
        "X": X,
        "y": y,
        "xx": xx,
        "yy": yy,
        "confidence": confidence,
        "far": far[0],
        "far_guess": far_guess,
        "far_confidence": float(far_proba.max()),
    }
