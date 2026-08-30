"""
Synthetic machine-telemetry generator for the AI Predictive Maintenance System.

Design goals
------------
1. SHORT.  1,500 readings instead of a 10,000-row reference corpus, so the file
   stays readable, ships inside the repository and trains in a second.

2. COLLINEARITY PRESERVED.  A naive generator draws every sensor column
   independently, which destroys the physical structure of real industrial
   telemetry and produces a model whose feature importances are meaningless.
   This generator keeps the three real dependencies intact:

   a) Thermal coupling      process_temperature = air_temperature + ~10 K
      Ambient heat propagates into the process, so the two temperature
      channels move together (expected Pearson r ~ +0.85).

   b) Mechanical coupling   rotational_speed = 60 * P / (2*pi*torque)
      Speed and torque are two views of the same delivered power, so they are
      inversely related (expected Pearson r ~ -0.75).

   c) Duty coupling         tool wear accumulates faster on light-duty (L)
      units and slower on heavy-duty (H) units, tying a numeric channel to the
      categorical quality channel.

3. LABELS ARE A FUNCTION OF THE FEATURES.  The failure flag is produced by
   five deterministic physical failure modes (tool wear, heat dissipation,
   power, overstrain, random), never by an independent coin flip.  That is what
   makes torque, rotational speed and tool wear emerge as the dominant
   predictors instead of noise.

Run:  python scripts/generate_synthetic_dataset.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Failure-mode thresholds (kept in one place so they can be audited)
# ---------------------------------------------------------------------------
# Failure-mode thresholds.  Each mode is a graded hazard rather than a hard
# cut-off: risk ramps up through a transition band centred on the threshold.
# Real equipment does not fail the instant a limit is crossed, and a step
# function would train a classifier that only ever answers 0% or 100%, leaving
# the intermediate health bands (Good / Warning) permanently empty.
TOOL_WEAR_THRESHOLD_MIN = 208.0
TOOL_WEAR_RAMP = 8.0

HEAT_MIN_DELTA_K = 8.8                  # below this the unit cannot shed heat
HEAT_DELTA_RAMP = 0.34
HEAT_MAX_RPM = 1410                     # ... and airflow is too low to help
HEAT_RPM_RAMP = 45.0

POWER_MIN_W = 3700.0
POWER_MAX_W = 9100.0
POWER_RAMP = 150.0

OVERSTRAIN_LIMIT = {"L": 9500.0, "M": 10500.0, "H": 11500.0}   # min*Nm
OVERSTRAIN_RAMP = 480.0

OVERSPEED_RPM = 2545.0                  # bearing overspeed
OVERSPEED_RAMP = 48.0

TORQUE_OVERLOAD_NM = 56.2               # drive-train continuous rating
# A wider ramp here on purpose: torque is the strongest single predictor, and a
# knife-edge transition would collapse the intermediate risk bands into a
# near-binary decision.
TORQUE_OVERLOAD_RAMP = 3.0

# Ceiling on any single mode, so no reading is ever a guaranteed failure.
MODE_CEILING = 0.90
RANDOM_FAILURE_RATE = 0.004

# Share of readings sampled from a stressed mechanical envelope.  Without this a
# short 1,500-row file would contain only ~50 positives, too few to train on.
# Stress is applied to the LOAD channels only (torque, power, wear) and never to
# ambient temperature, otherwise air temperature would become a proxy for the
# label rather than an honest sensor reading.
STRESSED_SHARE = 0.30

# A separate, smaller population of readings with a degrading cooling circuit.
# This is the only mechanism that compresses the air/process temperature gap and
# is what makes heat-dissipation failures possible.
COOLING_FAULT_SHARE = 0.07


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _sample_types(rng: np.random.Generator, rows: int) -> np.ndarray:
    mix = config.DATASET["type_mix"]
    labels = list(mix.keys())
    weights = np.array([mix[label] for label in labels], dtype=float)
    weights /= weights.sum()
    return rng.choice(labels, size=rows, p=weights)


def _ambient_walk(rng: np.random.Generator, rows: int) -> np.ndarray:
    """Ambient temperature as a mean-reverting random walk, not white noise.

    Real plant ambient conditions drift over a shift; consecutive readings are
    correlated.  A pure normal draw would wipe that structure out.
    """
    cfg = config.DATASET
    walk = np.zeros(rows)
    value = 0.0
    for index in range(rows):
        value = 0.94 * value + rng.normal(0.0, 0.75)
        walk[index] = value
    walk *= cfg["air_temp_sd"] / max(walk.std(), 1e-9)
    return cfg["air_temp_mean"] + walk


def generate_frame() -> pd.DataFrame:
    cfg = config.DATASET
    rows = cfg["rows"]
    rng = np.random.default_rng(cfg["random_seed"])

    machine_type = _sample_types(rng, rows)
    stressed = rng.random(rows) < STRESSED_SHARE
    cooling_fault = rng.random(rows) < COOLING_FAULT_SHARE

    # --- (a) thermal channels: collinear pair --------------------------------
    # Ambient temperature is deliberately independent of machine stress.
    air_temp = _ambient_walk(rng, rows)

    offset = rng.normal(cfg["process_offset_mean"], cfg["process_offset_sd"], rows)
    # A degrading cooling circuit compresses the air/process gap, which is the
    # physical precursor of a heat-dissipation failure.
    offset -= np.where(cooling_fault, rng.gamma(2.6, 1.15, rows), 0.0)
    process_temp = air_temp + offset

    # --- (b) mechanical channels: inversely collinear pair -------------------
    torque = rng.normal(cfg["torque_mean"], cfg["torque_sd"] * 1.1, rows)
    torque += np.where(stressed, rng.normal(0.0, 13.0, rows), 0.0)
    torque = np.clip(torque, 3.0, 77.0)

    power = rng.normal(cfg["target_power_w_mean"] * 2.1,
                       cfg["target_power_w_sd"] * 7.6, rows)
    power += np.where(stressed, rng.normal(0.0, 1750.0, rows), 0.0)
    power = np.clip(power, 2600.0, 10600.0)

    # Speed follows from delivered power and torque: the inverse relationship
    # between torque and rpm is created here, not imposed afterwards.
    rpm = power * 60.0 / (2.0 * np.pi * torque)
    rpm = np.clip(rpm, 1168.0, 2886.0)
    # Re-derive power from the clipped speed so the columns stay consistent.
    power = torque * rpm * 2.0 * np.pi / 60.0

    # --- (c) duty-coupled tool wear -----------------------------------------
    wear_rate = np.select(
        [machine_type == "L", machine_type == "M", machine_type == "H"],
        [1.18, 1.00, 0.84],
        default=1.0,
    )
    base_wear = rng.gamma(2.1, 30.0, rows)
    base_wear += np.where(stressed, rng.gamma(2.2, 24.0, rows), 0.0)
    tool_wear = np.clip(np.round(base_wear * wear_rate), 0, cfg["tool_wear_max"])

    thermal_delta = process_temp - air_temp

    # --- failure modes, as graded hazards ------------------------------------
    limits = np.array([OVERSTRAIN_LIMIT[t] for t in machine_type], dtype=float)

    # TWF - tool wear failure
    p_twf = MODE_CEILING * _sigmoid(
        (tool_wear - TOOL_WEAR_THRESHOLD_MIN) / TOOL_WEAR_RAMP
    )
    # HDF - heat dissipation failure: needs a collapsed thermal gap AND low airflow
    p_hdf = MODE_CEILING * _sigmoid(
        (HEAT_MIN_DELTA_K - thermal_delta) / HEAT_DELTA_RAMP
    ) * _sigmoid((HEAT_MAX_RPM - rpm) / HEAT_RPM_RAMP)
    # PWF - power failure at either end of the envelope
    p_pwf = MODE_CEILING * np.clip(
        _sigmoid((POWER_MIN_W - power) / POWER_RAMP)
        + _sigmoid((power - POWER_MAX_W) / POWER_RAMP),
        0.0,
        1.0,
    )
    # OSF - overstrain from combined wear and load
    p_osf = MODE_CEILING * _sigmoid((tool_wear * torque - limits) / OVERSTRAIN_RAMP)
    # OSP - bearing overspeed
    p_osp = MODE_CEILING * _sigmoid((rpm - OVERSPEED_RPM) / OVERSPEED_RAMP)
    # TOL - torque overload
    p_tol = MODE_CEILING * _sigmoid(
        (torque - TORQUE_OVERLOAD_NM) / TORQUE_OVERLOAD_RAMP
    )
    # RNF - random, unexplained failures
    p_rnf = np.full(rows, RANDOM_FAILURE_RATE)

    modes = np.vstack([p_twf, p_hdf, p_pwf, p_osf, p_osp, p_tol, p_rnf])
    # Competing independent risks: survival is the product of not failing.
    hazard = 1.0 - np.prod(1.0 - modes, axis=0)

    failure = (rng.random(rows) < hazard).astype(int)

    # Per-mode flags mark which hazards were materially active for a reading,
    # used only for reporting - the label itself comes from the hazard above.
    twf = (p_twf > 0.5).astype(int)
    hdf = (p_hdf > 0.5).astype(int)
    pwf = (p_pwf > 0.5).astype(int)
    osf = (p_osf > 0.5).astype(int)
    osp = (p_osp > 0.5).astype(int)
    tol = (p_tol > 0.5).astype(int)
    rnf = np.zeros(rows, dtype=int)
    dominant_index = np.argmax(modes, axis=0)
    dominant = np.array(
        ["twf", "hdf", "pwf", "osf", "osp", "tol", "rnf"]
    )[dominant_index]

    # --- identifiers --------------------------------------------------------
    machine_ids = [m["id"] for m in config.MACHINES]
    machine_by_type: dict[str, list[str]] = {}
    for machine in config.MACHINES:
        machine_by_type.setdefault(machine["type"], []).append(machine["id"])

    assigned = []
    for t in machine_type:
        pool = machine_by_type.get(t) or machine_ids
        assigned.append(pool[int(rng.integers(0, len(pool)))])

    start = datetime(2025, 5, 4, 6, 0, 0)
    timestamps = [
        (start + timedelta(hours=6 * index)).strftime("%Y-%m-%d %H:%M:%S")
        for index in range(rows)
    ]

    product_serial = 47000 + np.arange(rows)
    product_id = [f"{t}{serial}" for t, serial in zip(machine_type, product_serial)]

    frame = pd.DataFrame(
        {
            "udi": np.arange(1, rows + 1),
            "timestamp": timestamps,
            "machine_id": assigned,
            "product_id": product_id,
            "machine_type": machine_type,
            "air_temperature_k": np.round(air_temp, 1),
            "process_temperature_k": np.round(process_temp, 1),
            "rotational_speed_rpm": np.round(rpm).astype(int),
            "torque_nm": np.round(torque, 1),
            "tool_wear_min": tool_wear.astype(int),
            "thermal_delta_k": np.round(thermal_delta, 2),
            "power_w": np.round(power, 1),
            "hazard": np.round(hazard, 4),
            "dominant_mode": dominant,
            "machine_failure": failure,
            "twf": twf.astype(int),
            "hdf": hdf.astype(int),
            "pwf": pwf.astype(int),
            "osf": osf.astype(int),
            "osp": osp.astype(int),
            "tol": tol.astype(int),
            "rnf": rnf.astype(int),
        }
    )
    return frame


# ---------------------------------------------------------------------------
# Collinearity audit
# ---------------------------------------------------------------------------
def variance_inflation_factors(frame: pd.DataFrame,
                               columns: list[str]) -> dict[str, float]:
    """VIF via ordinary least squares R-squared, using numpy only.

    VIF = 1 / (1 - R^2) where R^2 comes from regressing one predictor on the
    remaining predictors.  A value above ~5 confirms the column shares
    information with its neighbours, i.e. collinearity is present.
    """
    matrix = frame[columns].to_numpy(dtype=float)
    result: dict[str, float] = {}
    for index, name in enumerate(columns):
        target = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        predicted = design @ coefficients
        residual_ss = float(((target - predicted) ** 2).sum())
        total_ss = float(((target - target.mean()) ** 2).sum())
        r_squared = 1.0 - residual_ss / total_ss if total_ss else 0.0
        result[name] = round(1.0 / max(1.0 - r_squared, 1e-9), 2)
    return result


def build_profile(frame: pd.DataFrame) -> dict:
    numeric = config.NUMERIC_FEATURES
    correlation = frame[numeric].corr(method="pearson").round(3)

    profile = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": int(len(frame)),
        "positive_rows": int(frame["machine_failure"].sum()),
        "positive_rate_pct": round(
            float(frame["machine_failure"].mean() * 100.0), 2
        ),
        "columns": list(frame.columns),
        "model_features": numeric + config.CATEGORICAL_FEATURES,
        "correlation_labels": [config.FEATURE_LABELS[c] for c in numeric],
        "correlation_keys": numeric,
        "correlation_matrix": correlation.values.tolist(),
        "vif": variance_inflation_factors(frame, numeric),
        "preserved_relationships": [
            {
                "name": "Thermal coupling",
                "pair": "Air Temperature <-> Process Temp",
                "expected": "strong positive",
                "observed_r": round(
                    float(frame["air_temperature_k"].corr(
                        frame["process_temperature_k"])), 3
                ),
            },
            {
                "name": "Mechanical coupling",
                "pair": "Torque <-> Rotational Speed",
                "expected": "strong negative",
                "observed_r": round(
                    float(frame["torque_nm"].corr(
                        frame["rotational_speed_rpm"])), 3
                ),
            },
            {
                "name": "Duty coupling",
                "pair": "Machine Type <-> Tool Wear",
                "expected": "light-duty wears faster",
                "observed_r": round(
                    float(
                        frame["tool_wear_min"].corr(
                            frame["machine_type"].map({"L": 2, "M": 1, "H": 0})
                        )
                    ),
                    3,
                ),
            },
        ],
        "failure_modes": {
            "twf_tool_wear": int(frame["twf"].sum()),
            "hdf_heat_dissipation": int(frame["hdf"].sum()),
            "pwf_power": int(frame["pwf"].sum()),
            "osf_overstrain": int(frame["osf"].sum()),
            "osp_overspeed": int(frame["osp"].sum()),
            "tol_torque_overload": int(frame["tol"].sum()),
        },
        "dominant_mode_counts": {
            key: int(value)
            for key, value in frame.loc[
                frame["machine_failure"] == 1, "dominant_mode"
            ].value_counts().items()
        },
        "hazard": {
            "mean": round(float(frame["hazard"].mean()), 4),
            "median": round(float(frame["hazard"].median()), 4),
            "graded_share_pct": round(
                float(
                    ((frame["hazard"] > 0.1) & (frame["hazard"] < 0.9)).mean() * 100
                ),
                1,
            ),
        },
        "type_mix": {
            key: int(value)
            for key, value in frame["machine_type"].value_counts().items()
        },
    }
    return profile


def main() -> None:
    frame = generate_frame()
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.TELEMETRY_CSV, index=False)

    profile = build_profile(frame)
    profile_path = config.DATASET_DIR / "dataset_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print("=" * 78)
    print("  SYNTHETIC MACHINE TELEMETRY - GENERATED")
    print("=" * 78)
    print(f"  File            : {config.TELEMETRY_CSV}")
    print(f"  Rows            : {profile['rows']}")
    print(f"  Failure rows    : {profile['positive_rows']} "
          f"({profile['positive_rate_pct']}%)")
    print(f"  Machine mix     : {profile['type_mix']}")
    print()
    print("  COLLINEARITY PRESERVED")
    print("  " + "-" * 74)
    for item in profile["preserved_relationships"]:
        print(f"  {item['name']:<20} {item['pair']:<38} r = {item['observed_r']:+.3f}")
    print()
    print("  Variance Inflation Factors (>5 = shared information retained)")
    for name, value in profile["vif"].items():
        print(f"    {config.FEATURE_LABELS[name]:<26} VIF = {value}")
    print()
    print("  Hazard modes active (p > 0.5)")
    for name, value in profile["failure_modes"].items():
        print(f"    {name:<24} {value}")
    print()
    print("  Dominant mode among failures")
    for name, value in profile["dominant_mode_counts"].items():
        print(f"    {name:<24} {value}")
    print()
    print(f"  Mean hazard      : {profile['hazard']['mean']}")
    print(f"  Graded readings  : {profile['hazard']['graded_share_pct']}% "
          f"sit between 10% and 90% risk")
    print()
    print("  Hazard calibration (binned hazard vs observed failure rate)")
    print("  " + "-" * 74)
    bins = [0.0, 0.1, 0.25, 0.45, 0.65, 0.85, 1.01]
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (frame["hazard"] >= low) & (frame["hazard"] < high)
        count = int(mask.sum())
        if count == 0:
            continue
        observed = float(frame.loc[mask, "machine_failure"].mean()) * 100
        print(f"    hazard {low:.2f}-{high:.2f}   n={count:>4}   "
              f"observed failures {observed:>5.1f}%")
    print()
    print(f"  Profile written  : {profile_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
