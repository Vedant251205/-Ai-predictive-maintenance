"""
Kiln stoppage analytics.

Reads the stoppage history once, caches it, and derives the availability
metrics the Kiln Stoppage Analytics page reports: downtime, MTBF, MTTR,
availability, cause Pareto, monthly trend and shift distribution.
"""

from __future__ import annotations

import threading
from datetime import datetime

import pandas as pd

import config

_lock = threading.Lock()
_frame: pd.DataFrame | None = None


class KilnDataUnavailable(RuntimeError):
    pass


def _window() -> tuple[datetime, datetime, int, float]:
    start = datetime.strptime(config.KILN["start_date"], "%Y-%m-%d")
    end = datetime.strptime(config.KILN["end_date"], "%Y-%m-%d")
    days = (end - start).days + 1          # inclusive of both end dates
    return start, end, days, days * 24.0


def load() -> pd.DataFrame:
    global _frame
    if _frame is None:
        with _lock:
            if _frame is None:
                if not config.KILN_CSV.exists():
                    raise KilnDataUnavailable(
                        "dataset/kiln_stoppages.csv not found. Run: "
                        "python scripts/generate_kiln_dataset.py"
                    )
                frame = pd.read_csv(config.KILN_CSV)
                frame["start_dt"] = pd.to_datetime(frame["start_time"])
                frame["end_dt"] = pd.to_datetime(frame["end_time"])
                frame["month_label"] = frame["start_dt"].dt.strftime("%b %Y")
                _frame = frame
    return _frame


def reload() -> None:
    global _frame
    with _lock:
        _frame = None


def kpis() -> dict:
    frame = load()
    _, _, days, total_hours = _window()

    stoppages = int(len(frame))
    downtime = float(frame["duration_hours"].sum())
    uptime = total_hours - downtime
    unplanned = frame[frame["planned"] == 0]
    planned = frame[frame["planned"] == 1]

    return {
        "unit": config.KILN["unit"],
        "organisation": config.BRAND["organisation"],
        "dataset_file": config.KILN_CSV.name,
        "start_date": datetime.strptime(
            config.KILN["start_date"], "%Y-%m-%d"
        ).strftime("%d-%m-%Y"),
        "end_date": datetime.strptime(
            config.KILN["end_date"], "%Y-%m-%d"
        ).strftime("%d-%m-%Y"),
        "days": days,
        "total_hours": round(total_hours, 1),
        "stoppages": stoppages,
        "downtime_hours": round(downtime, 1),
        "uptime_hours": round(uptime, 1),
        "availability": round(uptime / total_hours * 100.0, 1),
        # MTBF as calendar period over stoppage count, the convention used on
        # the plant availability sheet.  The uptime-based figure is published
        # alongside it so the definition is never ambiguous.
        "mtbf": round(total_hours / max(stoppages, 1), 1),
        "mtbf_uptime_basis": round(uptime / max(stoppages, 1), 1),
        "mttr": round(downtime / max(stoppages, 1), 1),
        "longest_stoppage": round(float(frame["duration_hours"].max()), 1),
        "shortest_stoppage": round(float(frame["duration_hours"].min()), 1),
        "production_loss": round(float(frame["production_loss_tonnes"].sum()), 0),
        "unplanned_count": int(len(unplanned)),
        "unplanned_hours": round(float(unplanned["duration_hours"].sum()), 1),
        "planned_count": int(len(planned)),
        "planned_hours": round(float(planned["duration_hours"].sum()), 1),
        "avg_per_month": round(stoppages / max(days / 30.44, 1), 1),
    }


def cause_breakdown() -> list[dict]:
    """Pareto of downtime by cause, worst first."""
    frame = load()
    downtime = float(frame["duration_hours"].sum()) or 1.0

    grouped = (
        frame.groupby("cause_category")
        .agg(
            events=("stoppage_id", "count"),
            hours=("duration_hours", "sum"),
            loss=("production_loss_tonnes", "sum"),
        )
        .sort_values("hours", ascending=False)
        .reset_index()
    )

    cumulative = 0.0
    rows: list[dict] = []
    for _, row in grouped.iterrows():
        share = float(row["hours"]) / downtime * 100.0
        cumulative += share
        rows.append({
            "cause": row["cause_category"],
            "events": int(row["events"]),
            "hours": round(float(row["hours"]), 1),
            "share": round(share, 1),
            "cumulative": round(cumulative, 1),
            "loss": round(float(row["loss"]), 0),
            "mttr": round(float(row["hours"]) / max(int(row["events"]), 1), 1),
        })
    return rows


def section_breakdown() -> list[dict]:
    frame = load()
    grouped = (
        frame.groupby("section")
        .agg(events=("stoppage_id", "count"), hours=("duration_hours", "sum"))
        .sort_values("hours", ascending=False)
        .reset_index()
    )
    return [
        {
            "section": row["section"],
            "events": int(row["events"]),
            "hours": round(float(row["hours"]), 1),
        }
        for _, row in grouped.iterrows()
    ]


def department_breakdown() -> list[dict]:
    frame = load()
    grouped = (
        frame.groupby("responsible_department")
        .agg(events=("stoppage_id", "count"), hours=("duration_hours", "sum"))
        .sort_values("hours", ascending=False)
        .reset_index()
    )
    return [
        {
            "department": row["responsible_department"],
            "events": int(row["events"]),
            "hours": round(float(row["hours"]), 1),
        }
        for _, row in grouped.iterrows()
    ]


def monthly_trend() -> dict:
    """Stoppage count and downtime hours per calendar month."""
    frame = load().sort_values("start_dt")
    grouped = (
        frame.groupby(["month", "month_label"], sort=True)
        .agg(events=("stoppage_id", "count"), hours=("duration_hours", "sum"))
        .reset_index()
        .sort_values("month")
    )
    return {
        "labels": grouped["month_label"].tolist(),
        "events": [int(value) for value in grouped["events"]],
        "hours": [round(float(value), 1) for value in grouped["hours"]],
    }


def shift_breakdown() -> list[dict]:
    frame = load()
    grouped = (
        frame.groupby("shift")
        .agg(events=("stoppage_id", "count"), hours=("duration_hours", "sum"))
        .reset_index()
        .sort_values("shift")
    )
    return [
        {
            "shift": row["shift"],
            "events": int(row["events"]),
            "hours": round(float(row["hours"]), 1),
        }
        for _, row in grouped.iterrows()
    ]


def duration_histogram() -> dict:
    frame = load()
    edges = [0, 2, 4, 8, 16, 24, 48, 1000]
    labels = ["<2 h", "2-4 h", "4-8 h", "8-16 h", "16-24 h", "24-48 h", ">48 h"]
    counts = [0] * len(labels)
    for value in frame["duration_hours"]:
        for index in range(len(labels)):
            if edges[index] <= value < edges[index + 1]:
                counts[index] += 1
                break
    return {"labels": labels, "values": counts}


def longest_stoppages(limit: int = 8) -> list[dict]:
    frame = load().sort_values("duration_hours", ascending=False).head(limit)
    return [
        {
            "stoppage_id": row["stoppage_id"],
            "start_time": row["start_time"],
            "duration_hours": round(float(row["duration_hours"]), 1),
            "cause": row["cause_category"],
            "section": row["section"],
            "department": row["responsible_department"],
            "shift": row["shift"],
            "loss": round(float(row["production_loss_tonnes"]), 0),
            "planned": bool(row["planned"]),
        }
        for _, row in frame.iterrows()
    ]


def recent_stoppages(limit: int = 12) -> list[dict]:
    frame = load().sort_values("start_dt", ascending=False).head(limit)
    return [
        {
            "stoppage_id": row["stoppage_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration_hours": round(float(row["duration_hours"]), 1),
            "cause": row["cause_category"],
            "section": row["section"],
            "department": row["responsible_department"],
            "shift": row["shift"],
            "loss": round(float(row["production_loss_tonnes"]), 0),
            "planned": bool(row["planned"]),
        }
        for _, row in frame.iterrows()
    ]
