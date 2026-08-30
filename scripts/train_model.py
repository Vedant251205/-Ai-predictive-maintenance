"""
Trains the Random Forest predictive-maintenance classifier.

Pipeline
--------
    ColumnTransformer
        numeric      -> StandardScaler
        categorical  -> OneHotEncoder
    RandomForestClassifier(n_estimators=150)

Artefacts written to models/:
    rf_pipeline.joblib    fitted preprocessor + classifier
    model_metadata.json   metrics, feature importances, collinearity audit

Run:  python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


NUMERIC_SLICE = list(range(len(config.NUMERIC_FEATURES)))     # columns 0..4
CATEGORICAL_SLICE = [len(config.NUMERIC_FEATURES)]            # column 5


def build_pipeline() -> Pipeline:
    # Columns are selected by POSITION, not by name, so the fitted pipeline
    # accepts a plain NumPy array at inference time and pandas is not needed in
    # the deployed runtime. Categories are declared explicitly so the one-hot
    # block has a stable width even if a class is absent from a split.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_SLICE),
            (
                "cat",
                OneHotEncoder(
                    categories=[[0, 1, 2]],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_SLICE,
            ),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=config.ML["n_estimators"],
        max_depth=config.ML["max_depth"],
        min_samples_leaf=config.ML["min_samples_leaf"],
        random_state=config.ML["random_state"],
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def aggregate_importances(pipeline: Pipeline) -> dict[str, float]:
    """Collapse one-hot columns back onto their source feature, then scale to %.

    The transformer emits the scaled numeric columns first, in the order of
    NUMERIC_FEATURES, followed by the one-hot block for the machine class. The
    mapping is therefore positional, which stays correct now that the
    ColumnTransformer selects columns by index rather than by name.
    """
    raw = pipeline.named_steps["classifier"].feature_importances_
    numeric_count = len(config.NUMERIC_FEATURES)

    totals: dict[str, float] = {
        name: float(raw[index])
        for index, name in enumerate(config.NUMERIC_FEATURES)
    }
    # Everything after the numeric block belongs to the machine class.
    totals["machine_type"] = float(sum(raw[numeric_count:]))

    total = sum(totals.values()) or 1.0
    return {name: round(value / total * 100.0, 2) for name, value in totals.items()}


def build_matrix(frame) -> "np.ndarray":
    """Assemble the numeric design matrix the pipeline expects."""
    columns = [frame[name].to_numpy(dtype=float)
               for name in config.NUMERIC_FEATURES]
    codes = frame["machine_type"].map(config.MACHINE_TYPE_CODES)
    if codes.isna().any():
        raise ValueError("dataset contains an unknown machine_type value")
    columns.append(codes.to_numpy(dtype=float))
    return np.column_stack(columns)


def main() -> None:
    if not config.TELEMETRY_CSV.exists():
        raise SystemExit(
            "dataset/machine_telemetry.csv missing. "
            "Run: python scripts/generate_synthetic_dataset.py"
        )

    frame = pd.read_csv(config.TELEMETRY_CSV)
    features = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X = build_matrix(frame)
    y = frame[config.TARGET].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.ML["test_size"],
        random_state=config.ML["random_state"],
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predicted = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predicted)
    folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.ML["random_state"]
    )
    cv_scores = cross_val_score(build_pipeline(), X, y, cv=folds, scoring="f1")

    permutation = permutation_importance(
        pipeline, X_test, y_test, n_repeats=12,
        random_state=config.ML["random_state"], scoring="f1",
    )
    permutation_pct = {}
    total_perm = float(np.clip(permutation.importances_mean, 0, None).sum()) or 1.0
    for name, value in zip(features, permutation.importances_mean):
        permutation_pct[name] = round(max(float(value), 0.0) / total_perm * 100.0, 2)

    importances = aggregate_importances(pipeline)
    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)

    profile_path = config.DATASET_DIR / "dataset_profile.json"
    profile = (
        json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.exists()
        else {}
    )

    metadata = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "RandomForestClassifier",
        "n_estimators": config.ML["n_estimators"],
        "max_depth": config.ML["max_depth"],
        "preprocessor": "ColumnTransformer(StandardScaler + OneHotEncoder)",
        "features": features,
        "feature_labels": {k: config.FEATURE_LABELS[k] for k in features},
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_pct": round(float(y.mean() * 100.0), 2),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predicted)) * 100, 2),
            "precision": round(float(precision_score(y_test, predicted, zero_division=0)) * 100, 2),
            "recall": round(float(recall_score(y_test, predicted, zero_division=0)) * 100, 2),
            "f1": round(float(f1_score(y_test, predicted, zero_division=0)) * 100, 2),
            "roc_auc": round(float(roc_auc_score(y_test, probability)) * 100, 2),
            "cv_f1_mean": round(float(cv_scores.mean()) * 100, 2),
            "cv_f1_std": round(float(cv_scores.std()) * 100, 2),
        },
        "confusion_matrix": {
            "true_negative": int(matrix[0][0]),
            "false_positive": int(matrix[0][1]),
            "false_negative": int(matrix[1][0]),
            "true_positive": int(matrix[1][1]),
        },
        "feature_importance": importances,
        "feature_importance_ranked": [
            {"feature": name, "label": config.FEATURE_LABELS[name], "pct": value}
            for name, value in ranked
        ],
        "permutation_importance": permutation_pct,
        "collinearity": {
            "vif": profile.get("vif", {}),
            "correlation_keys": profile.get("correlation_keys", []),
            "correlation_labels": profile.get("correlation_labels", []),
            "correlation_matrix": profile.get("correlation_matrix", []),
            "preserved_relationships": profile.get("preserved_relationships", []),
        },
        "failure_modes": profile.get("failure_modes", {}),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    config.MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 78)
    print("  RANDOM FOREST TRAINING COMPLETE")
    print("=" * 78)
    print(f"  Rows            : {metadata['dataset_rows']} "
          f"(train {metadata['train_rows']} / test {metadata['test_rows']})")
    print(f"  Positive rate   : {metadata['positive_rate_pct']}%")
    print()
    for key, value in metadata["metrics"].items():
        print(f"  {key:<16}: {value}%")
    print()
    print("  Feature importance")
    print("  " + "-" * 74)
    for entry in metadata["feature_importance_ranked"]:
        bar = "#" * int(entry["pct"] / 1.5)
        print(f"    {entry['label']:<26} {entry['pct']:>6.2f}%  {bar}")
    print()
    print("  Sanity check on reference operating points")
    print("  " + "-" * 74)
    samples = [
        ("Nominal      ", "M", 300.0, 310.0, 1500, 40.0, 0),
        ("Mild wear    ", "M", 303.2, 314.7, 1497, 34.5, 39),
        ("Heavy wear   ", "H", 304.2, 319.4, 1912, 53.4, 94),
        ("Overstrain   ", "L", 302.0, 311.0, 1320, 62.0, 205),
        ("Heat stall   ", "M", 303.0, 309.5, 1290, 45.0, 120),
    ]
    for label, machine_type, air, process, rpm, torque, wear in samples:
        row = np.array([[
            air, process, rpm, torque, wear,
            config.MACHINE_TYPE_CODES[machine_type],
        ]], dtype=float)
        probability_value = float(pipeline.predict_proba(row)[0][1]) * 100.0
        print(f"    {label} fail = {probability_value:6.2f}%   "
              f"health = {100 - probability_value:6.2f}")
    print()
    print(f"  Model     : {config.MODEL_PATH}")
    print(f"  Metadata  : {config.MODEL_META_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
