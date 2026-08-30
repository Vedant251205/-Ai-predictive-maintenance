"""
Kiln stoppage analytics.

Reads the precomputed analytics produced by
scripts/precompute_kiln_analytics.py and serves it. The figures are derived from
a static dataset by fixed formulas, so they are computed once at development time
rather than recalculated from the CSV on every request.

That choice removes pandas from the request path. pandas is roughly 62 MB
installed, and dropping it is what brings the deployed bundle under a serverless
size limit. Regenerating the source data is a two-step operation:

    python scripts/generate_kiln_dataset.py
    python scripts/precompute_kiln_analytics.py
"""

from __future__ import annotations

import json
import threading

import config

_lock = threading.Lock()
_data: dict | None = None

ANALYTICS_PATH = config.DATASET_DIR / "kiln_analytics.json"


class KilnDataUnavailable(RuntimeError):
    pass


def load() -> dict:
    """Load and cache the precomputed analytics block."""
    global _data
    if _data is None:
        with _lock:
            if _data is None:
                if not ANALYTICS_PATH.exists():
                    raise KilnDataUnavailable(
                        "dataset/kiln_analytics.json not found. Run: "
                        "python scripts/generate_kiln_dataset.py && "
                        "python scripts/precompute_kiln_analytics.py"
                    )
                _data = json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    return _data


def reload() -> None:
    global _data
    with _lock:
        _data = None


def kpis() -> dict:
    return load()["kpis"]


def cause_breakdown() -> list[dict]:
    """Pareto of downtime by cause, worst first."""
    return load()["causes"]


def section_breakdown() -> list[dict]:
    return load()["sections"]


def department_breakdown() -> list[dict]:
    return load()["departments"]


def monthly_trend() -> dict:
    """Stoppage count and downtime hours per calendar month."""
    return load()["monthly"]


def shift_breakdown() -> list[dict]:
    return load()["shifts"]


def duration_histogram() -> dict:
    return load()["duration_histogram"]


def longest_stoppages(limit: int = 8) -> list[dict]:
    return load()["longest"][:limit]


def recent_stoppages(limit: int = 12) -> list[dict]:
    return load()["recent"][:limit]


def chronological_stoppages() -> list[dict]:
    """Stoppages in start order, used to verify none of them overlap."""
    return load()["chronological"]
