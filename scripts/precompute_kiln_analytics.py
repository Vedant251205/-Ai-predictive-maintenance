"""
Precomputes every kiln analytics figure into dataset/kiln_analytics.json.

Why: the kiln module originally derived its metrics from the CSV with pandas on
every request. pandas is roughly 62 MB installed, which is the difference
between fitting inside a serverless size limit and not. The maths is fixed and
the source data is static, so it is computed once here, at development time, and
the application simply reads the result.

Run this whenever the kiln dataset is regenerated:

    python scripts/generate_kiln_dataset.py
    python scripts/precompute_kiln_analytics.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OUTPUT = config.DATASET_DIR / "kiln_analytics.json"

SHIFT_ORDER = ["A (06:00-14:00)", "B (14:00-22:00)", "C (22:00-06:00)"]
HISTOGRAM_EDGES = [0, 2, 4, 8, 16, 24, 48, 10**9]
HISTOGRAM_LABELS = ["<2 h", "2-4 h", "4-8 h", "8-16 h", "16-24 h",
                    "24-48 h", ">48 h"]


def window() -> tuple[datetime, datetime, int, float]:
    start = datetime.strptime(config.KILN["start_date"], "%Y-%m-%d")
    end = datetime.strptime(config.KILN["end_date"], "%Y-%m-%d")
    days = (end - start).days + 1          # inclusive of both end dates
    return start, end, days, days * 24.0


def build(frame: pd.DataFrame) -> dict:
    start, end, days, total_hours = window()

    frame = frame.copy()
    frame["start_dt"] = pd.to_datetime(frame["start_time"])
    frame["end_dt"] = pd.to_datetime(frame["end_time"])
    frame["month_label"] = frame["start_dt"].dt.strftime("%b %Y")

    stoppages = int(len(frame))
    downtime = float(frame["duration_hours"].sum())
    uptime = total_hours - downtime
    unplanned = frame[frame["planned"] == 0]
    planned = frame[frame["planned"] == 1]

    kpis = {
        "unit": config.KILN["unit"],
        "organisation": config.BRAND["organisation"],
        "dataset_file": config.KILN_CSV.name,
        "start_date": start.strftime("%d-%m-%Y"),
        "end_date": end.strftime("%d-%m-%Y"),
        "days": days,
        "total_hours": round(total_hours, 1),
        "stoppages": stoppages,
        "downtime_hours": round(downtime, 1),
        "uptime_hours": round(uptime, 1),
        "availability": round(uptime / total_hours * 100.0, 1),
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

    # Cause Pareto, worst first, with a running cumulative share.
    grouped = (
        frame.groupby("cause_category")
        .agg(events=("stoppage_id", "count"),
             hours=("duration_hours", "sum"),
             loss=("production_loss_tonnes", "sum"))
        .sort_values("hours", ascending=False)
        .reset_index()
    )
    causes = []
    cumulative = 0.0
    for _, row in grouped.iterrows():
        share = float(row["hours"]) / (downtime or 1.0) * 100.0
        cumulative += share
        causes.append({
            "cause": row["cause_category"],
            "events": int(row["events"]),
            "hours": round(float(row["hours"]), 1),
            "share": round(share, 1),
            "cumulative": round(cumulative, 1),
            "loss": round(float(row["loss"]), 0),
            "mttr": round(float(row["hours"]) / max(int(row["events"]), 1), 1),
        })

    def simple_group(column, key):
        block = (
            frame.groupby(column)
            .agg(events=("stoppage_id", "count"),
                 hours=("duration_hours", "sum"))
            .sort_values("hours", ascending=False)
            .reset_index()
        )
        return [{key: row[column],
                 "events": int(row["events"]),
                 "hours": round(float(row["hours"]), 1)}
                for _, row in block.iterrows()]

    shifts_block = (
        frame.groupby("shift")
        .agg(events=("stoppage_id", "count"), hours=("duration_hours", "sum"))
        .reset_index()
    )
    shifts_block["order"] = shifts_block["shift"].apply(
        lambda value: SHIFT_ORDER.index(value) if value in SHIFT_ORDER else 99
    )
    shifts_block = shifts_block.sort_values("order")
    shifts = [{"shift": row["shift"],
               "events": int(row["events"]),
               "hours": round(float(row["hours"]), 1)}
              for _, row in shifts_block.iterrows()]

    monthly_block = (
        frame.groupby(["month", "month_label"], sort=True)
        .agg(events=("stoppage_id", "count"), hours=("duration_hours", "sum"))
        .reset_index()
        .sort_values("month")
    )
    monthly = {
        "labels": monthly_block["month_label"].tolist(),
        "events": [int(v) for v in monthly_block["events"]],
        "hours": [round(float(v), 1) for v in monthly_block["hours"]],
    }

    counts = [0] * len(HISTOGRAM_LABELS)
    for value in frame["duration_hours"]:
        for index in range(len(HISTOGRAM_LABELS)):
            if HISTOGRAM_EDGES[index] <= value < HISTOGRAM_EDGES[index + 1]:
                counts[index] += 1
                break
    histogram = {"labels": HISTOGRAM_LABELS, "values": counts}

    def rows_of(subset):
        return [{
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
        } for _, row in subset.iterrows()]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kpis": kpis,
        "causes": causes,
        "sections": simple_group("section", "section"),
        "departments": simple_group("responsible_department", "department"),
        "shifts": shifts,
        "monthly": monthly,
        "duration_histogram": histogram,
        "longest": rows_of(
            frame.sort_values("duration_hours", ascending=False).head(12)),
        "recent": rows_of(
            frame.sort_values("start_dt", ascending=False).head(20)),
        # Chronological rows, used by the test that asserts stoppages never
        # overlap without needing pandas at test time.
        "chronological": [
            {"stoppage_id": row["stoppage_id"],
             "start_time": row["start_time"],
             "end_time": row["end_time"]}
            for _, row in frame.sort_values("start_dt").iterrows()
        ],
    }


def main() -> None:
    if not config.KILN_CSV.exists():
        raise SystemExit(
            "dataset/kiln_stoppages.csv missing. Run: "
            "python scripts/generate_kiln_dataset.py"
        )

    frame = pd.read_csv(config.KILN_CSV)
    payload = build(frame)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    kpis = payload["kpis"]
    print("=" * 70)
    print("  KILN ANALYTICS PRECOMPUTED")
    print("=" * 70)
    print(f"  File          : {OUTPUT}")
    print(f"  Window        : {kpis['days']} days")
    print(f"  Stoppages     : {kpis['stoppages']}")
    print(f"  Downtime      : {kpis['downtime_hours']} h")
    print(f"  Availability  : {kpis['availability']}%")
    print(f"  MTBF / MTTR   : {kpis['mtbf']} h / {kpis['mttr']} h")
    print(f"  Cause groups  : {len(payload['causes'])}")
    print(f"  Size          : {OUTPUT.stat().st_size:,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    main()
