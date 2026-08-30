"""
Predictive cockpit: sensor entry, model inference and the result card.
"""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for

import config
from routes.auth import actor_label, login_required
from services import database, fleet_service, ml_service, prediction_service
from utils import validators

bp = Blueprint("predict", __name__)


def _defaults() -> dict:
    values = {
        field: bounds["default"]
        for field, bounds in config.OPERATING_RANGES.items()
    }
    values["machine_type"] = "L"
    values["machine_id"] = ""
    values["runtime_hours"] = 2000
    return values


@bp.route("/predict", methods=["GET", "POST"])
@login_required
def cockpit():
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    result = None
    form_values = _defaults()

    preset_key = request.args.get("preset")
    if request.method == "GET" and preset_key in prediction_service.PRESETS:
        form_values.update(prediction_service.PRESETS[preset_key]["values"])
        flash(
            f"Loaded the '{prediction_service.PRESETS[preset_key]['label']}' "
            "preset. Review the values and run the analysis.",
            "info",
        )

    machine_key = request.args.get("machine_id", "").upper()
    if request.method == "GET" and machine_key in config.MACHINE_INDEX:
        live = fleet_service.machine(machine_key)
        if live:
            form_values.update({
                "machine_id": live["machine_id"],
                "machine_type": live["machine_type"],
                "air_temperature_k": live["air_temperature_k"],
                "process_temperature_k": live["process_temperature_k"],
                "rotational_speed_rpm": live["rotational_speed_rpm"],
                "torque_nm": live["torque_nm"],
                "tool_wear_min": live["tool_wear_min"],
                "runtime_hours": live["runtime_hours"],
            })
            flash(
                f"Loaded the current live reading for {machine_key}.", "info"
            )

    if request.method == "POST":
        cleaned, errors = validators.validate_sensor_form(request.form)
        form_values.update(
            {key: request.form.get(key, value)
             for key, value in form_values.items()}
        )

        if errors:
            for message in errors:
                flash(message, "danger")
        else:
            try:
                result = prediction_service.assess(cleaned)
            except ml_service.ModelUnavailable as error:
                flash(str(error), "danger")
            else:
                form_values.update(cleaned)
                record_id = database.save_prediction(result, actor_label())
                result["record_id"] = record_id
                database.log_action(
                    actor_label(),
                    "prediction",
                    f"{result['machine_id'] or 'ad-hoc'} -> "
                    f"{result['status']} (health {result['health_score']}, "
                    f"priority {result['priority']})",
                )

                if result["status"] in ("Warning", "Critical"):
                    severity = (
                        "critical" if result["status"] == "Critical" else "warning"
                    )
                    machine_ref = result["machine_id"] or "AD-HOC"
                    if not database.recent_alert_exists(machine_ref, severity, 10):
                        database.add_alert(
                            machine_ref,
                            severity,
                            f"{machine_ref} {result['status'].upper()} - "
                            f"health {result['health_score']}",
                            result["recommendation"]["body"],
                            channel="prediction",
                        )

    return render_template(
        "predict.html",
        page_title="AI Prediction",
        ranges=config.OPERATING_RANGES,
        labels=config.FEATURE_LABELS,
        machine_types=config.MACHINE_TYPES,
        machines=config.MACHINES,
        presets=prediction_service.PRESETS,
        values=form_values,
        result=result,
        fleet=fleet,
        summary=summary,
    )
