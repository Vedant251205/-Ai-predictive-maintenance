"""
Machine learning service.

Owns the trained Random Forest pipeline: loading it once, exposing failure
probabilities, and publishing the training metadata (metrics, feature
importances, collinearity audit) that the Analytics and AI Feature Intelligence
pages render.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading

import joblib
import pandas as pd

import config

_lock = threading.Lock()
_pipeline = None
_metadata: dict | None = None

FEATURE_ORDER = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


class ModelUnavailable(RuntimeError):
    """Raised when the serialised pipeline cannot be found or loaded."""


def artefacts_present() -> bool:
    return config.MODEL_PATH.exists() and config.TELEMETRY_CSV.exists()


def bootstrap(verbose: bool = True) -> None:
    """Generate the datasets and train the model if artefacts are missing.

    Keeps first-run friction at zero: `python app.py` on a clean clone produces
    a working platform without any manual preparation step.
    """
    scripts_dir = config.BASE_DIR / "scripts"
    steps = []
    if not config.TELEMETRY_CSV.exists():
        steps.append(scripts_dir / "generate_synthetic_dataset.py")
    if not config.KILN_CSV.exists():
        steps.append(scripts_dir / "generate_kiln_dataset.py")
    if not config.MODEL_PATH.exists() or not config.MODEL_META_PATH.exists():
        steps.append(scripts_dir / "train_model.py")

    for script in steps:
        if verbose:
            print(f"[bootstrap] running {script.name} ...")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(config.BASE_DIR),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ModelUnavailable(
                f"{script.name} failed:\n{completed.stdout}\n{completed.stderr}"
            )


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                if not config.MODEL_PATH.exists():
                    raise ModelUnavailable(
                        "models/rf_pipeline.joblib not found. Run: "
                        "python scripts/train_model.py"
                    )
                _pipeline = joblib.load(config.MODEL_PATH)
    return _pipeline


def get_metadata() -> dict:
    global _metadata
    if _metadata is None:
        with _lock:
            if _metadata is None:
                if config.MODEL_META_PATH.exists():
                    _metadata = json.loads(
                        config.MODEL_META_PATH.read_text(encoding="utf-8")
                    )
                else:
                    _metadata = {}
    return _metadata


def reload() -> None:
    """Drop cached artefacts so the next call re-reads from disk."""
    global _pipeline, _metadata
    with _lock:
        _pipeline = None
        _metadata = None


def _frame(readings: list[dict]) -> pd.DataFrame:
    rows = []
    for reading in readings:
        rows.append({
            "air_temperature_k": float(reading["air_temperature_k"]),
            "process_temperature_k": float(reading["process_temperature_k"]),
            "rotational_speed_rpm": float(reading["rotational_speed_rpm"]),
            "torque_nm": float(reading["torque_nm"]),
            "tool_wear_min": float(reading["tool_wear_min"]),
            "machine_type": str(reading.get("machine_type", "M")).upper(),
        })
    return pd.DataFrame(rows)[FEATURE_ORDER]


def failure_probability(reading: dict) -> float:
    """Probability of failure as a percentage, 0-100."""
    pipeline = get_pipeline()
    proba = pipeline.predict_proba(_frame([reading]))[0][1]
    return round(float(proba) * 100.0, 2)


def failure_probabilities(readings: list[dict]) -> list[float]:
    if not readings:
        return []
    pipeline = get_pipeline()
    proba = pipeline.predict_proba(_frame(readings))[:, 1]
    return [round(float(value) * 100.0, 2) for value in proba]


# ---------------------------------------------------------------------------
# Published metadata
# ---------------------------------------------------------------------------
def metrics() -> dict:
    return get_metadata().get("metrics", {})


def confusion() -> dict:
    return get_metadata().get("confusion_matrix", {})


def feature_importance() -> list[dict]:
    return get_metadata().get("feature_importance_ranked", [])


def permutation_importance() -> dict:
    return get_metadata().get("permutation_importance", {})


def collinearity() -> dict:
    return get_metadata().get("collinearity", {})


def failure_modes() -> dict:
    return get_metadata().get("failure_modes", {})


def model_card() -> dict:
    meta = get_metadata()
    return {
        "algorithm": meta.get("algorithm", "RandomForestClassifier"),
        "n_estimators": meta.get("n_estimators", config.ML["n_estimators"]),
        "max_depth": meta.get("max_depth", config.ML["max_depth"]),
        "preprocessor": meta.get(
            "preprocessor", "ColumnTransformer(StandardScaler + OneHotEncoder)"
        ),
        "dataset_rows": meta.get("dataset_rows", 0),
        "train_rows": meta.get("train_rows", 0),
        "test_rows": meta.get("test_rows", 0),
        "positive_rate_pct": meta.get("positive_rate_pct", 0),
        "trained_at": meta.get("trained_at", "-"),
        "features": meta.get("features", FEATURE_ORDER),
        "feature_labels": meta.get("feature_labels", config.FEATURE_LABELS),
    }
