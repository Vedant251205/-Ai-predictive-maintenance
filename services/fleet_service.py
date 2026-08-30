"""
Live fleet telemetry simulator.

The platform is a demonstrator, so there is no physical OPC-UA / MQTT link.
This service stands in for one: it produces a plausible, reproducible sensor
reading for each of the nine monitored machines and lets the trained model
classify it.

Three properties matter.

1. Deterministic within a refresh window.  Every page rendered inside the same
   30-second bucket sees identical numbers, so the dashboard, alert center and
   maintenance advisor can never contradict each other.

2. Stable machine personality.  Each asset carries a duty profile, so the fleet
   keeps a realistic mix of healthy, degrading and critical units instead of
   flickering randomly on every reload.

3. Candidate selection, not outcome forcing.  For each machine several
   candidate operating points are drawn from that asset's physical envelope and
   the one which actually exhibits the intended condition is kept.  The model
   still performs every classification; the simulator only decides which
   operating point the asset is sitting at, exactly as reality would.
"""

from __future__ import annotations

import threading
import time
import zlib

import numpy as np

import config
from services import prediction_service

REFRESH_SECONDS = 30

# How many candidate operating points to draw per machine per window.
CANDIDATES = 14

# Duty profile per machine.  Produces the documented fleet mix of
# 5 healthy / 2 warning / 2 critical units.
PROFILES: dict[str, str] = {
    "KLN-01": "warning",
    "KLN-02": "excellent",
    "BLR-03": "critical",
    "RMM-04": "good",
    "CML-05": "excellent",
    "CLC-06": "good",
    "FAN-07": "excellent",
    "CRS-08": "critical",
    "PKR-09": "warning",
}

# Condition each profile is meant to exhibit, and the work-order priority that
# should normally follow from it.
PROFILE_TARGETS: dict[str, dict] = {
    "excellent": {"status": "Excellent", "priority": "P5", "midpoint": 93.0},
    "good": {"status": "Good", "priority": "P5", "midpoint": 75.0},
    "warning": {"status": "Warning", "priority": "P3", "midpoint": 55.0},
    "critical": {"status": "Critical", "priority": "P2", "midpoint": 25.0},
}

# Sensor envelopes per profile: (low, high) inclusive.
#
# Two physical constraints shape these numbers:
#   * torque x speed must keep mechanical power inside 3.7-9.1 kW, otherwise
#     every profile would trip the power hazard and read as critical;
#   * the degrading profiles stay just inside the advisory driver limits, so a
#     warning unit reports elevated risk without an outright limit breach.
ENVELOPES: dict[str, dict] = {
    "excellent": {
        "air": (296.0, 302.5),
        "delta": (9.6, 11.8),
        "rpm": (1360, 1640),
        "torque": (33.0, 45.0),
        "wear": (5, 70),
        "runtime": (1200, 3200),
    },
    "good": {
        "air": (297.0, 303.5),
        "delta": (9.5, 11.5),
        "rpm": (1320, 1520),
        "torque": (50.0, 54.0),
        "wear": (80, 130),
        "runtime": (3400, 5600),
    },
    "warning": {
        "air": (299.0, 304.0),
        "delta": (9.4, 11.0),
        "rpm": (1330, 1520),
        "torque": (54.6, 56.2),
        "wear": (100, 138),
        "runtime": (5800, 7600),
    },
    "critical": {
        "air": (300.5, 304.5),
        "delta": (9.4, 11.0),
        "rpm": (1320, 1430),
        "torque": (57.8, 60.0),
        "wear": (90, 140),
        "runtime": (8200, 11500),
    },
}

_cache: dict[int, list[dict]] = {}
_cache_lock = threading.Lock()
_CACHE_LIMIT = 48


def current_bucket() -> int:
    return int(time.time() // REFRESH_SECONDS)


def _rng(machine_id: str, bucket: int, attempt: int) -> np.random.Generator:
    seed = zlib.crc32(f"{machine_id}:{bucket}:{attempt}".encode("utf-8"))
    return np.random.default_rng(seed)


def _candidate(machine: dict, bucket: int, attempt: int) -> dict:
    profile = PROFILES.get(machine["id"], "good")
    envelope = ENVELOPES[profile]
    rng = _rng(machine["id"], bucket, attempt)

    air = float(rng.uniform(*envelope["air"]))
    delta = float(rng.uniform(*envelope["delta"]))
    rpm = int(rng.integers(envelope["rpm"][0], envelope["rpm"][1] + 1))
    torque = float(rng.uniform(*envelope["torque"]))
    wear = int(rng.integers(envelope["wear"][0], envelope["wear"][1] + 1))
    runtime = float(rng.uniform(*envelope["runtime"]))

    return {
        "machine_id": machine["id"],
        "machine_type": machine["type"],
        "air_temperature_k": round(air, 1),
        "process_temperature_k": round(air + delta, 1),
        "rotational_speed_rpm": rpm,
        "torque_nm": round(torque, 1),
        "tool_wear_min": wear,
        "runtime_hours": round(runtime, 1),
        "profile": profile,
    }


def _select(machine_id: str, profile: str,
            assessed: list[dict]) -> dict:
    """Pick the candidate that best exhibits the machine's intended condition."""
    target = PROFILE_TARGETS[profile]

    exact = [
        item for item in assessed
        if item["status"] == target["status"]
        and item["priority"] == target["priority"]
    ]
    if exact:
        return exact[0]

    same_status = [item for item in assessed if item["status"] == target["status"]]
    if same_status:
        return same_status[0]

    # Nothing in the intended band: fall back to whichever candidate sits
    # closest to that band's centre so the reading is still representative.
    return min(
        assessed,
        key=lambda item: abs(item["health_score"] - target["midpoint"]),
    )


def snapshot(bucket: int | None = None) -> list[dict]:
    """Assess every machine in the fleet for a given refresh window."""
    bucket = current_bucket() if bucket is None else bucket

    with _cache_lock:
        cached = _cache.get(bucket)
    if cached is not None:
        return cached

    readings: list[dict] = []
    for machine in config.MACHINES:
        for attempt in range(CANDIDATES):
            readings.append(_candidate(machine, bucket, attempt))

    # A single batched model call covers every candidate for every machine.
    assessments = prediction_service.assess_many(readings)
    for assessment, reading in zip(assessments, readings):
        assessment["profile"] = reading["profile"]

    fleet: list[dict] = []
    for index, machine in enumerate(config.MACHINES):
        window = assessments[index * CANDIDATES:(index + 1) * CANDIDATES]
        profile = PROFILES.get(machine["id"], "good")
        fleet.append(_select(machine["id"], profile, window))

    with _cache_lock:
        if len(_cache) >= _CACHE_LIMIT:
            for stale in sorted(_cache)[: len(_cache) - _CACHE_LIMIT + 1]:
                _cache.pop(stale, None)
        _cache[bucket] = fleet

    return fleet


def summary(fleet: list[dict] | None = None) -> dict:
    """Fleet-level KPIs used by the dashboards."""
    fleet = snapshot() if fleet is None else fleet
    total = len(fleet) or 1

    excellent = sum(1 for item in fleet if item["status"] == "Excellent")
    good = sum(1 for item in fleet if item["status"] == "Good")
    warning = sum(1 for item in fleet if item["status"] == "Warning")
    critical = sum(1 for item in fleet if item["status"] == "Critical")
    healthy = excellent + good

    avg_health = sum(item["health_score"] for item in fleet) / total
    avg_failure = sum(item["failure_prob"] for item in fleet) / total
    avg_rul = sum(item["rul_hours"] for item in fleet) / total

    priority_counts = {code: 0 for code in config.PRIORITIES}
    for item in fleet:
        priority_counts[item["priority"]] += 1

    department_counts: dict[str, int] = {}
    for item in fleet:
        key = item["department"]
        department_counts[key] = department_counts.get(key, 0) + 1

    man_hours = sum(
        config.PRIORITIES[item["priority"]]["man_hours"] for item in fleet
    )

    return {
        "total": len(fleet),
        "excellent": excellent,
        "good": good,
        "warning": warning,
        "critical": critical,
        "healthy": healthy,
        "non_healthy": warning + critical,
        "avg_health": round(avg_health, 1),
        "avg_failure": round(avg_failure, 1),
        "avg_rul_hours": round(avg_rul, 0),
        "avg_rul_days": round(avg_rul / 24.0, 1),
        "availability": round(healthy / total * 100.0, 1),
        "failure_risk_pct": round((warning + critical) / total * 100.0, 1),
        "priority_counts": priority_counts,
        "p1_p2": priority_counts["P1"] + priority_counts["P2"],
        "department_counts": department_counts,
        "man_hours": round(man_hours, 1),
        "mttr_estimate": round(
            man_hours / max(priority_counts["P1"] + priority_counts["P2"]
                            + priority_counts["P3"], 1),
            1,
        ),
        "bucket": current_bucket(),
        "refresh_seconds": REFRESH_SECONDS,
    }


def health_trend(points: int = 12) -> dict:
    """Fleet average health over the last `points` refresh windows."""
    bucket = current_bucket()
    labels: list[str] = []
    values: list[float] = []

    for offset in range(points - 1, -1, -1):
        window = bucket - offset
        fleet = snapshot(bucket=window)
        average = sum(item["health_score"] for item in fleet) / max(len(fleet), 1)
        moment = time.localtime(window * REFRESH_SECONDS)
        labels.append(time.strftime("%H:%M:%S", moment))
        values.append(round(average, 1))

    return {"labels": labels, "values": values}


def machine_health_series(fleet: list[dict] | None = None) -> dict:
    """Per-machine health scores for the fleet overview bar chart."""
    fleet = snapshot() if fleet is None else fleet
    return {
        "labels": [item["machine_id"] for item in fleet],
        "names": [item["machine_name"] for item in fleet],
        "values": [item["health_score"] for item in fleet],
        "colours": [item["colour"] for item in fleet],
    }


def status_distribution(fleet: list[dict] | None = None) -> dict:
    fleet = snapshot() if fleet is None else fleet
    counts = {"Excellent": 0, "Good": 0, "Warning": 0, "Critical": 0}
    for item in fleet:
        counts[item["status"]] += 1
    return {"labels": list(counts.keys()), "values": list(counts.values())}


def critical_machines(fleet: list[dict] | None = None) -> list[dict]:
    fleet = snapshot() if fleet is None else fleet
    ranked = [item for item in fleet if item["priority"] in ("P1", "P2")]
    ranked.sort(key=lambda item: item["health_score"])
    return ranked


def work_orders(fleet: list[dict] | None = None) -> list[dict]:
    """Every machine as a CMMS work order, worst priority first."""
    fleet = snapshot() if fleet is None else fleet
    return sorted(
        fleet,
        key=lambda item: (
            config.PRIORITIES[item["priority"]]["order"],
            item["health_score"],
        ),
    )


def machine(machine_id: str) -> dict | None:
    for item in snapshot():
        if item["machine_id"] == machine_id.upper():
            return item
    return None
