"""
Advisory engine.

Turns a raw model output plus the sensor reading that produced it into the
human-facing artefacts the platform displays: the driver breakdown, the AI
recommendation paragraph, a CMMS priority code and a corrective action plan.
"""

from __future__ import annotations

import math

import config

TWO_PI_OVER_60 = 2.0 * math.pi / 60.0

# Thresholds mirrored from the physical failure modes encoded in the dataset
# generator, so an explanation never contradicts what the model learned.  Each
# limit sits marginally outside the corresponding hazard midpoint used during
# generation, so a reading that is merely approaching a limit raises elevated
# risk without being reported as an outright breach.
OVERSPEED_RPM = 2570            # hazard midpoint 2545
UNDERSPEED_RPM = 1300
TORQUE_OVERLOAD_NM = 56.5       # hazard midpoint 56.2
TORQUE_UNDERLOAD_NM = 18.0
POWER_MIN_W = 3600.0            # hazard midpoint 3700
POWER_MAX_W = 9200.0            # hazard midpoint 9100
WEAR_CRITICAL_MIN = 200         # hazard midpoint 208
WEAR_WARNING_MIN = 150
OVERSTRAIN_LIMIT = {"L": 9500, "M": 10500, "H": 11500}


def mechanical_power_w(torque_nm: float, rpm: float) -> float:
    """P = torque * angular velocity, the quantity the power failure mode uses."""
    return float(torque_nm) * float(rpm) * TWO_PI_OVER_60


def analyse_drivers(reading: dict) -> list[dict]:
    """Explain which sensor channels are pushing the risk up.

    Returns an ordered list, worst first, of {label, severity, message, value}.
    """
    air = float(reading["air_temperature_k"])
    process = float(reading["process_temperature_k"])
    rpm = float(reading["rotational_speed_rpm"])
    torque = float(reading["torque_nm"])
    wear = float(reading["tool_wear_min"])
    delta = process - air
    power = mechanical_power_w(torque, rpm)

    drivers: list[dict] = []

    if delta < config.ALERT_THRESHOLDS["min_thermal_delta"]:
        drivers.append({
            "label": "Heat dissipation",
            "severity": "critical",
            "value": f"dT {delta:.1f} K",
            "message": (
                f"Air-to-process gap has collapsed to {delta:.1f} K, below the "
                f"{config.ALERT_THRESHOLDS['min_thermal_delta']} K minimum. The "
                "unit can no longer shed heat."
            ),
        })
    elif delta < 9.5:
        drivers.append({
            "label": "Heat dissipation",
            "severity": "warning",
            "value": f"dT {delta:.1f} K",
            "message": f"Cooling margin is thin at {delta:.1f} K. Inspect the "
                       "cooling circuit before the next shift.",
        })

    if air > 305.0:
        drivers.append({
            "label": "Ambient temperature",
            "severity": "warning",
            "value": f"{air:.1f} K",
            "message": f"Ambient is {air:.1f} K, above the 305 K comfort "
                       "threshold, which reduces available cooling.",
        })

    if process > 315.0:
        drivers.append({
            "label": "Process temperature",
            "severity": "warning",
            "value": f"{process:.1f} K",
            "message": f"Process temperature {process:.1f} K exceeds the 315 K "
                       "normal band.",
        })

    if rpm >= OVERSPEED_RPM:
        drivers.append({
            "label": "Bearing overspeed",
            "severity": "critical",
            "value": f"{rpm:.0f} RPM",
            "message": f"Drive is turning at {rpm:.0f} RPM, past the "
                       f"{OVERSPEED_RPM} RPM overspeed limit. Bearing damage is "
                       "likely.",
        })
    elif rpm < UNDERSPEED_RPM:
        drivers.append({
            "label": "Low speed",
            "severity": "warning",
            "value": f"{rpm:.0f} RPM",
            "message": f"Speed {rpm:.0f} RPM is below the rated 1300 RPM "
                       "envelope, which also weakens forced airflow.",
        })

    if torque >= TORQUE_OVERLOAD_NM:
        drivers.append({
            "label": "Torque overload",
            "severity": "critical",
            "value": f"{torque:.1f} Nm",
            "message": f"Load of {torque:.1f} Nm is beyond the "
                       f"{TORQUE_OVERLOAD_NM} Nm continuous rating of the drive "
                       "train.",
        })
    elif torque < TORQUE_UNDERLOAD_NM:
        drivers.append({
            "label": "Low torque",
            "severity": "warning",
            "value": f"{torque:.1f} Nm",
            "message": f"Torque {torque:.1f} Nm is unusually low; check for "
                       "slippage or an unloaded drive.",
        })

    if wear >= WEAR_CRITICAL_MIN:
        drivers.append({
            "label": "Tool wear",
            "severity": "critical",
            "value": f"{wear:.0f} min",
            "message": f"Wear parts have run {wear:.0f} min, past the "
                       f"{WEAR_CRITICAL_MIN} min replacement point.",
        })
    elif wear >= WEAR_WARNING_MIN:
        drivers.append({
            "label": "Tool wear",
            "severity": "warning",
            "value": f"{wear:.0f} min",
            "message": f"Wear parts at {wear:.0f} min. Schedule replacement "
                       f"before {WEAR_CRITICAL_MIN} min.",
        })

    overstrain = wear * torque
    limit = OVERSTRAIN_LIMIT.get(
        str(reading.get("machine_type", "M")).upper(), 10500
    )
    if overstrain > limit:
        drivers.append({
            "label": "Overstrain",
            "severity": "critical",
            "value": f"{overstrain:,.0f} min*Nm",
            "message": f"Combined wear and torque product {overstrain:,.0f} "
                       f"exceeds the {limit:,} limit for this machine class.",
        })

    if power > POWER_MAX_W:
        drivers.append({
            "label": "Power draw",
            "severity": "critical",
            "value": f"{power:,.0f} W",
            "message": f"Delivered power {power:,.0f} W is above the "
                       f"{POWER_MAX_W:,.0f} W ceiling.",
        })
    elif power < POWER_MIN_W:
        drivers.append({
            "label": "Power draw",
            "severity": "critical",
            "value": f"{power:,.0f} W",
            "message": f"Delivered power {power:,.0f} W has fallen below the "
                       f"{POWER_MIN_W:,.0f} W floor; the drive is stalling.",
        })

    order = {"critical": 0, "warning": 1, "info": 2}
    drivers.sort(key=lambda item: order.get(item["severity"], 3))
    return drivers


def recommendation(status: str, drivers: list[dict]) -> dict:
    """The AI Recommendation panel content."""
    critical = [item for item in drivers if item["severity"] == "critical"]
    warning = [item for item in drivers if item["severity"] == "warning"]

    if status == "Critical":
        headline = "Immediate intervention required"
        body = (
            "The model predicts a high probability of failure. "
            + " ".join(item["message"] for item in critical[:2])
            + " Stop the unit at the next safe opportunity and raise an "
              "emergency work order."
        )
    elif status == "Warning":
        headline = "Degradation detected"
        body = (
            "Failure risk is rising above the acceptable band. "
            + " ".join(item["message"] for item in (critical + warning)[:2])
            + " Plan corrective maintenance within the current week."
        )
    elif status == "Good":
        if warning or critical:
            body = (
                "Overall condition is acceptable but not optimal. "
                + " ".join(item["message"] for item in (critical + warning)[:1])
                + " Add the unit to the next scheduled inspection round."
            )
        else:
            body = (
                "Condition is acceptable. No parameter is outside its operating "
                "envelope. Continue the normal inspection cycle."
            )
        headline = "Stable with minor deviations"
    else:
        headline = "Optimal operating conditions"
        body = (
            "Optimal operational conditions. The ML model predicts no failure "
            "risks. Continue standard operations."
        )

    band = next(
        (entry for entry in config.HEALTH_BANDS if entry[1] == status),
        config.HEALTH_BANDS[-1],
    )
    return {
        "headline": headline,
        "body": body,
        "action": band[3],
        "colour": band[2],
        "critical_count": len(critical),
        "warning_count": len(warning),
    }


def assign_priority(status: str, failure_prob: float,
                    drivers: list[dict]) -> str:
    """Map a prediction onto a P1-P5 CMMS work-order priority."""
    critical_drivers = sum(1 for item in drivers if item["severity"] == "critical")

    if status == "Critical":
        if failure_prob >= 80.0 or critical_drivers >= 2:
            return "P1"
        return "P2"
    if status == "Warning":
        return "P2" if critical_drivers else "P3"
    if status == "Good":
        return "P4" if critical_drivers or failure_prob >= 30.0 else "P5"
    return "P5"


def action_plan(machine: dict, prediction: dict,
                drivers: list[dict]) -> dict:
    """Corrective and preventive steps for the Maintenance Advisor card."""
    corrective: list[str] = []
    preventive: list[str] = []

    labels = {item["label"] for item in drivers}

    if "Heat dissipation" in labels:
        corrective.append("Flush and pressure-test the cooling circuit.")
        preventive.append("Add a weekly cooling-margin trend check.")
    if "Bearing overspeed" in labels:
        corrective.append("Trim the drive set-point back inside 1300-1700 RPM.")
        preventive.append("Enable a hard speed interlock at 2500 RPM.")
    if "Low speed" in labels:
        corrective.append("Verify VFD reference signal and belt tension.")
    if "Torque overload" in labels or "Overstrain" in labels:
        corrective.append("Reduce feed rate and inspect the gearbox for backlash.")
        preventive.append("Log torque trend against feed rate each shift.")
    if "Tool wear" in labels:
        corrective.append("Replace wear liners and reset the wear counter.")
        preventive.append("Move wear-part replacement to a 180 min interval.")
    if "Power draw" in labels:
        corrective.append("Check motor current balance and coupling alignment.")
    if "Ambient temperature" in labels or "Process temperature" in labels:
        preventive.append("Review bay ventilation and fan damper positions.")

    if not corrective:
        corrective.append("No corrective work required at this reading.")
    if not preventive:
        preventive.append("Continue the standard preventive schedule.")

    priority = prediction["priority"]
    meta = config.PRIORITIES[priority]

    return {
        "priority": priority,
        "priority_label": meta["label"],
        "priority_colour": meta["colour"],
        "sla": meta["sla"],
        "man_hours": meta["man_hours"],
        "department": machine.get("department", "Mechanical"),
        "corrective": corrective,
        "preventive": preventive,
    }
