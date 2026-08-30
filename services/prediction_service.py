"""
Prediction service.

Converts a raw sensor reading into the complete assessment the platform shows:

    failure probability   <- Random Forest
    health score          = 100 - failure probability
    risk band             <- config.HEALTH_BANDS
    remaining useful life <- multi-factor degradation heuristic
    next service          <- band ceiling capped by RUL
    priority + advice     <- utils.recommendations

Remaining Useful Life is deliberately a documented heuristic rather than a
model output: the classifier answers "will this fail", not "when".  RUL is
composed of three multiplicative terms applied to a baseline service life.

    RUL = base_life * wear_term * health_term * stress_term

    wear_term    consumes life in proportion to elapsed tool wear
    health_term  scales life by the model's confidence in the asset
    stress_term  penalises operation away from the nominal duty point
"""

from __future__ import annotations

from datetime import datetime, timedelta

import config
from services import ml_service
from utils import recommendations


def health_band(health_score: float) -> tuple[str, str, str]:
    """Return (status, colour token, recommended action) for a health score."""
    for lower_bound, status, colour, action in config.HEALTH_BANDS:
        if health_score >= lower_bound:
            return status, colour, action
    last = config.HEALTH_BANDS[-1]
    return last[1], last[2], last[3]


def remaining_useful_life(reading: dict, failure_prob: float) -> dict:
    """Multi-factor RUL estimate in operating hours, with its own breakdown."""
    cfg = config.RUL

    tool_wear = float(reading["tool_wear_min"])
    torque = float(reading["torque_nm"])
    rpm = float(reading["rotational_speed_rpm"])
    delta = float(reading["process_temperature_k"]) - float(
        reading["air_temperature_k"]
    )

    wear_fraction = min(tool_wear / cfg["tool_wear_max"], 1.0)
    wear_term = 1.0 - cfg["wear_weight"] * wear_fraction

    health_fraction = 1.0 - min(max(failure_prob, 0.0), 100.0) / 100.0
    health_term = cfg["health_floor"] + (1.0 - cfg["health_floor"]) * health_fraction

    nominal_delta = cfg["thermal_nominal_delta"]
    thermal_high = max((delta - nominal_delta) / nominal_delta, 0.0)
    thermal_low = max((nominal_delta - delta) / nominal_delta, 0.0) * cfg[
        "thermal_low_multiplier"
    ]
    thermal_dev = max(thermal_high, thermal_low)

    speed_dev = max((rpm - cfg["speed_nominal_rpm"]) / cfg["speed_span_rpm"], 0.0)
    torque_dev = max((torque - cfg["torque_nominal_nm"]) / cfg["torque_span_nm"], 0.0)

    stress_penalty = (
        cfg["thermal_weight"] * thermal_dev
        + cfg["speed_weight"] * speed_dev
        + cfg["torque_weight"] * torque_dev
    )
    stress_term = max(1.0 - stress_penalty, cfg["stress_floor"])

    hours = cfg["base_life_hours"] * wear_term * health_term * stress_term
    hours = max(round(hours), 0)

    return {
        "hours": hours,
        "days": round(hours / 24.0, 1),
        "terms": {
            "wear": round(wear_term, 3),
            "health": round(health_term, 3),
            "stress": round(stress_term, 3),
        },
        "deviations": {
            "thermal": round(thermal_dev, 3),
            "speed": round(speed_dev, 3),
            "torque": round(torque_dev, 3),
        },
    }


def assess(reading: dict, machine: dict | None = None) -> dict:
    """Full assessment for one sensor reading."""
    machine_id = str(reading.get("machine_id") or "").upper()
    if machine is None:
        machine = config.MACHINE_INDEX.get(machine_id)

    machine_type = str(
        reading.get("machine_type")
        or (machine or {}).get("type")
        or "M"
    ).upper()

    payload = {
        "air_temperature_k": float(reading["air_temperature_k"]),
        "process_temperature_k": float(reading["process_temperature_k"]),
        "rotational_speed_rpm": int(round(float(reading["rotational_speed_rpm"]))),
        "torque_nm": round(float(reading["torque_nm"]), 1),
        "tool_wear_min": int(round(float(reading["tool_wear_min"]))),
        "machine_type": machine_type,
    }

    failure_prob = ml_service.failure_probability(payload)
    health_score = round(100.0 - failure_prob, 1)
    status, colour, action = health_band(health_score)

    drivers = recommendations.analyse_drivers(payload)
    advice = recommendations.recommendation(status, drivers)
    priority = recommendations.assign_priority(status, failure_prob, drivers)

    rul = remaining_useful_life(payload, failure_prob)

    band_ceiling = config.NEXT_SERVICE_DAYS.get(status, 30)
    next_service_days = max(1, min(band_ceiling, int(rul["days"])))
    next_service_date = (
        datetime.now() + timedelta(days=next_service_days)
    ).strftime("%d %b %Y")

    thermal_delta = round(
        payload["process_temperature_k"] - payload["air_temperature_k"], 2
    )
    power_w = round(
        recommendations.mechanical_power_w(
            payload["torque_nm"], payload["rotational_speed_rpm"]
        ),
        1,
    )

    assessment = {
        **payload,
        "machine_id": machine_id,
        "machine_name": (machine or {}).get("name", "Unassigned Asset"),
        "machine_icon": (machine or {}).get("icon", "fa-microchip"),
        "department": (machine or {}).get("department", "Operations"),
        "category": (machine or {}).get("category", "General"),
        "runtime_hours": round(float(reading.get("runtime_hours") or 0.0), 1),
        "failure_prob": failure_prob,
        "health_score": health_score,
        "status": status,
        "risk_level": status,
        "colour": colour,
        "action": advice["action"] or action,
        "priority": priority,
        "priority_label": config.PRIORITIES[priority]["label"],
        "priority_colour": config.PRIORITIES[priority]["colour"],
        "rul_hours": rul["hours"],
        "rul_days": rul["days"],
        "rul_terms": rul["terms"],
        "rul_deviations": rul["deviations"],
        "next_service_days": next_service_days,
        "next_service_date": next_service_date,
        "thermal_delta_k": thermal_delta,
        "power_w": power_w,
        "drivers": drivers,
        "recommendation": advice,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    assessment["work_order"] = recommendations.action_plan(
        machine or {}, assessment, drivers
    )
    return assessment


def assess_many(readings: list[dict]) -> list[dict]:
    """Vectorised assessment: one model call for the whole batch."""
    if not readings:
        return []

    payloads = []
    for reading in readings:
        machine_id = str(reading.get("machine_id") or "").upper()
        machine = config.MACHINE_INDEX.get(machine_id)
        payloads.append({
            "air_temperature_k": float(reading["air_temperature_k"]),
            "process_temperature_k": float(reading["process_temperature_k"]),
            "rotational_speed_rpm": int(round(float(reading["rotational_speed_rpm"]))),
            "torque_nm": round(float(reading["torque_nm"]), 1),
            "tool_wear_min": int(round(float(reading["tool_wear_min"]))),
            "machine_type": str(
                reading.get("machine_type") or (machine or {}).get("type") or "M"
            ).upper(),
        })

    probabilities = ml_service.failure_probabilities(payloads)

    results = []
    for reading, payload, failure_prob in zip(readings, payloads, probabilities):
        machine_id = str(reading.get("machine_id") or "").upper()
        machine = config.MACHINE_INDEX.get(machine_id) or {}

        health_score = round(100.0 - failure_prob, 1)
        status, colour, action = health_band(health_score)
        drivers = recommendations.analyse_drivers(payload)
        advice = recommendations.recommendation(status, drivers)
        priority = recommendations.assign_priority(status, failure_prob, drivers)
        rul = remaining_useful_life(payload, failure_prob)
        band_ceiling = config.NEXT_SERVICE_DAYS.get(status, 30)
        next_service_days = max(1, min(band_ceiling, int(rul["days"])))

        entry = {
            **payload,
            "machine_id": machine_id,
            "machine_name": machine.get("name", "Unassigned Asset"),
            "machine_icon": machine.get("icon", "fa-microchip"),
            "department": machine.get("department", "Operations"),
            "category": machine.get("category", "General"),
            "runtime_hours": round(float(reading.get("runtime_hours") or 0.0), 1),
            "failure_prob": failure_prob,
            "health_score": health_score,
            "status": status,
            "risk_level": status,
            "colour": colour,
            "action": advice["action"] or action,
            "priority": priority,
            "priority_label": config.PRIORITIES[priority]["label"],
            "priority_colour": config.PRIORITIES[priority]["colour"],
            "rul_hours": rul["hours"],
            "rul_days": rul["days"],
            "next_service_days": next_service_days,
            "thermal_delta_k": round(
                payload["process_temperature_k"] - payload["air_temperature_k"], 2
            ),
            "power_w": round(
                recommendations.mechanical_power_w(
                    payload["torque_nm"], payload["rotational_speed_rpm"]
                ),
                1,
            ),
            "drivers": drivers,
            "recommendation": advice,
            "created_at": reading.get(
                "created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
        }
        entry["work_order"] = recommendations.action_plan(machine, entry, drivers)
        results.append(entry)

    return results


PRESETS = {
    "healthy": {
        "label": "Healthy Operation",
        "risk": "Low Risk",
        "colour": "success",
        "icon": "fa-circle-check",
        "values": {
            "air_temperature_k": 298.5,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1480,
            "torque_nm": 38.5,
            "tool_wear_min": 24,
            "machine_type": "M",
            "runtime_hours": 2000,
        },
    },
    "warning": {
        "label": "Warning Alert",
        "risk": "Medium Risk",
        "colour": "warning",
        "icon": "fa-triangle-exclamation",
        "values": {
            "air_temperature_k": 302.4,
            "process_temperature_k": 312.1,
            "rotational_speed_rpm": 2080,
            "torque_nm": 54.6,
            "tool_wear_min": 148,
            "machine_type": "M",
            "runtime_hours": 6400,
        },
    },
    "critical": {
        "label": "Critical Failure",
        "risk": "High Risk",
        "colour": "danger",
        "icon": "fa-circle-exclamation",
        "values": {
            "air_temperature_k": 304.1,
            "process_temperature_k": 312.4,
            "rotational_speed_rpm": 2640,
            "torque_nm": 64.8,
            "tool_wear_min": 218,
            "machine_type": "L",
            "runtime_hours": 9100,
        },
    },
}
