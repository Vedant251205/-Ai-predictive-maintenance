"""
Synthetic kiln stoppage history for the Kiln Stoppage Analytics module.

The generator is calibrated so the derived KPIs land on the figures documented
for the platform:

    Analysis window   332 days  (2025-05-04 -> 2026-03-31) = 7,968 hours
    Total stoppages   51
    Total downtime    659.7 hours
    Kiln availability 91.7 %      (uptime / total period)
    MTBF              156.2 hours (total period / stoppages)
    MTTR              12.9 hours  (total downtime / stoppages)

Note on MTBF: the platform reports mean time BETWEEN failures as the calendar
period divided by the stoppage count, which is the convention used on the
plant's own availability sheets.  Dividing uptime instead of the full period
would give 143.3 hours.  Both numbers are exposed by the analytics service so
the definition is never ambiguous.

Run:  python scripts/generate_kiln_dataset.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


SECTIONS = [
    "Kiln Shell / Refractory",
    "Preheater Tower",
    "Calciner",
    "Clinker Cooler",
    "Burner / Firing System",
    "Kiln Main Drive",
    "Kiln Feed System",
    "ID Fan",
    "Instrumentation Loop",
]

# Cause -> (selection weight, mean duration hours, responsible department)
CAUSE_MODEL = {
    "Refractory / Coating Failure": (0.16, 34.0, "Mechanical"),
    "Mechanical Breakdown": (0.20, 16.0, "Mechanical"),
    "Electrical Fault": (0.15, 8.5, "Electrical"),
    "Instrumentation Fault": (0.13, 5.5, "Instrumentation"),
    "Fan / Blower Trip": (0.11, 7.0, "Electrical"),
    "Process Upset": (0.12, 4.5, "Operations"),
    "Raw Material Shortage": (0.06, 6.0, "Operations"),
    "Planned Preventive Halt": (0.07, 22.0, "Maintenance"),
}

MIN_GAP_HOURS = 20.0          # minimum idle gap enforced between stoppages
MIN_DURATION_HOURS = 0.6
MAX_DURATION_HOURS = 96.0
KILN_RATE_TPH = 145.8          # 3,500 tonnes/day clinker line


def _shift_for(hour: int) -> str:
    if 6 <= hour < 14:
        return "A (06:00-14:00)"
    if 14 <= hour < 22:
        return "B (14:00-22:00)"
    return "C (22:00-06:00)"


def _fit_durations(raw: np.ndarray, headroom: np.ndarray,
                   target_total: float) -> np.ndarray:
    """Scale durations onto an exact target total while respecting headroom.

    Each stoppage must finish before the next one starts, so every entry has an
    upper bound.  Scaling alone would violate those bounds, so the surplus is
    redistributed across the entries that still have room, repeatedly.
    """
    bounds = np.clip(headroom, MIN_DURATION_HOURS, MAX_DURATION_HOURS)
    values = np.clip(raw, MIN_DURATION_HOURS, bounds)

    for _ in range(400):
        deficit = target_total - values.sum()
        if abs(deficit) < 1e-6:
            break
        if deficit > 0:
            room = bounds - values
            available = room.sum()
            if available <= 1e-9:
                break
            values = values + room * (deficit / available)
        else:
            room = values - MIN_DURATION_HOURS
            available = room.sum()
            if available <= 1e-9:
                break
            values = values + room * (deficit / available)
        values = np.clip(values, MIN_DURATION_HOURS, bounds)

    return values


def generate_frame() -> pd.DataFrame:
    cfg = config.KILN
    rng = np.random.default_rng(cfg["random_seed"])

    start_date = datetime.strptime(cfg["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(cfg["end_date"], "%Y-%m-%d")
    # The analysis window is inclusive of both end dates.
    total_hours = ((end_date - start_date).days + 1) * 24.0
    count = cfg["target_stoppages"]

    # --- event start times: spread across the window, never bunched ----------
    slot = total_hours / count
    offsets = np.sort(
        np.arange(count) * slot + rng.uniform(0.0, slot * 0.82, count)
    )
    # Guarantee a minimum idle gap between consecutive events.
    for index in range(1, count):
        minimum = offsets[index - 1] + MIN_GAP_HOURS
        if offsets[index] < minimum:
            offsets[index] = minimum
    offsets = np.clip(offsets, 0.0, total_hours - MIN_GAP_HOURS)

    # --- causes -------------------------------------------------------------
    causes = list(CAUSE_MODEL.keys())
    weights = np.array([CAUSE_MODEL[c][0] for c in causes], dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(causes, size=count, p=weights)

    # --- raw durations, lognormal around each cause's mean -------------------
    means = np.array([CAUSE_MODEL[c][1] for c in chosen], dtype=float)
    raw = means * rng.lognormal(mean=-0.22, sigma=0.66, size=count)

    # Headroom = distance to the next stoppage (minus the enforced gap).
    next_offsets = np.append(offsets[1:], total_hours)
    headroom = next_offsets - offsets - MIN_GAP_HOURS

    durations = _fit_durations(raw, headroom, cfg["target_downtime_hours"])
    durations = np.round(durations, 2)
    # Absorb rounding drift into the entry with the most headroom.
    drift = round(cfg["target_downtime_hours"] - durations.sum(), 2)
    if abs(drift) >= 0.01:
        slack = np.clip(headroom, MIN_DURATION_HOURS, MAX_DURATION_HOURS) - durations
        durations[int(np.argmax(slack))] += drift
    durations = np.round(durations, 2)

    rows = []
    for index in range(count):
        start = start_date + timedelta(hours=float(offsets[index]))
        duration = float(durations[index])
        end = start + timedelta(hours=duration)
        cause = str(chosen[index])
        department = CAUSE_MODEL[cause][2]
        section = SECTIONS[int(rng.integers(0, len(SECTIONS)))]
        loss = duration * KILN_RATE_TPH * float(rng.uniform(0.88, 1.04))

        rows.append(
            {
                "stoppage_id": f"STP-{index + 1:03d}",
                "unit": cfg["unit"],
                "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_hours": round(duration, 2),
                "cause_category": cause,
                "section": section,
                "responsible_department": department,
                "shift": _shift_for(start.hour),
                "planned": int(cause == "Planned Preventive Halt"),
                "production_loss_tonnes": round(loss, 1),
                "month": start.strftime("%Y-%m"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = config.KILN
    frame = generate_frame()

    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.KILN_CSV, index=False)

    start_date = datetime.strptime(cfg["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(cfg["end_date"], "%Y-%m-%d")
    days = (end_date - start_date).days + 1
    total_hours = days * 24.0
    downtime = float(frame["duration_hours"].sum())
    uptime = total_hours - downtime
    stoppages = len(frame)

    print("=" * 78)
    print("  SYNTHETIC KILN STOPPAGE HISTORY - GENERATED")
    print("=" * 78)
    print(f"  File               : {config.KILN_CSV}")
    print(f"  Window             : {cfg['start_date']} -> {cfg['end_date']}")
    print(f"  Days analysed      : {days}          (target {cfg['days']})")
    print(f"  Total stoppages    : {stoppages}           "
          f"(target {cfg['target_stoppages']})")
    print(f"  Total downtime     : {downtime:.1f} h      "
          f"(target {cfg['target_downtime_hours']} h)")
    print(f"  Kiln availability  : {uptime / total_hours * 100:.1f} %")
    print(f"  MTBF (period/stop) : {total_hours / stoppages:.1f} h")
    print(f"  MTBF (uptime/stop) : {uptime / stoppages:.1f} h")
    print(f"  MTTR               : {downtime / stoppages:.1f} h")
    print(f"  Longest stoppage   : {frame['duration_hours'].max():.1f} h")
    print(f"  Production loss    : {frame['production_loss_tonnes'].sum():,.0f} t")
    print()
    print("  Downtime by cause")
    print("  " + "-" * 74)
    grouped = (
        frame.groupby("cause_category")["duration_hours"]
        .agg(["count", "sum"])
        .sort_values("sum", ascending=False)
    )
    for cause, row in grouped.iterrows():
        share = row["sum"] / downtime * 100
        print(f"    {cause:<32} {int(row['count']):>3} events  "
              f"{row['sum']:>7.1f} h  {share:>5.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
