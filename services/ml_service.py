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
import numpy as np

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
    kiln_analytics = config.DATASET_DIR / "kiln_analytics.json"

    model_ready = config.MODEL_PATH.exists() and config.MODEL_META_PATH.exists()
    kiln_ready = kiln_analytics.exists()

    # A slim deployment ships only the prebuilt artefacts: the trained model and
    # the precomputed kiln analytics. The raw CSVs and the generator scripts are
    # excluded to keep the bundle small. Nothing should be regenerated in that
    # case, and attempting it would fail because the scripts are absent.
    steps = []
    if not model_ready and not config.TELEMETRY_CSV.exists():
        steps.append(scripts_dir / "generate_synthetic_dataset.py")
    if not kiln_ready and not config.KILN_CSV.exists():
        steps.append(scripts_dir / "generate_kiln_dataset.py")
    if not model_ready:
        steps.append(scripts_dir / "train_model.py")
    if not kiln_ready:
        steps.append(scripts_dir / "precompute_kiln_analytics.py")

    missing = [step for step in steps if not step.exists()]
    if missing:
        raise ModelUnavailable(
            "Prebuilt artefacts are missing and the generator scripts are not "
            "present either: " + ", ".join(step.name for step in missing)
        )

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


def _matrix(readings: list[dict]) -> np.ndarray:
    """Build the numeric design matrix the fitted pipeline expects.

    Column order is fixed: the five numeric sensors, then the encoded machine
    class. A NumPy array is used rather than a DataFrame so pandas is not a
    runtime dependency, which keeps the deployed bundle small enough for a
    serverless host.
    """
    rows = []
    for reading in readings:
        machine_type = str(reading.get("machine_type", "M")).upper()
        rows.append([
            float(reading["air_temperature_k"]),
            float(reading["process_temperature_k"]),
            float(reading["rotational_speed_rpm"]),
            float(reading["torque_nm"]),
            float(reading["tool_wear_min"]),
            float(config.MACHINE_TYPE_CODES.get(machine_type, 1)),
        ])
    return np.asarray(rows, dtype=float)


def failure_probability(reading: dict) -> float:
    """Probability of failure as a percentage, 0-100."""
    pipeline = get_pipeline()
    proba = pipeline.predict_proba(_matrix([reading]))[0][1]
    return round(float(proba) * 100.0, 2)


def failure_probabilities(readings: list[dict]) -> list[float]:
    if not readings:
        return []
    pipeline = get_pipeline()
    proba = pipeline.predict_proba(_matrix(readings))[:, 1]
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
