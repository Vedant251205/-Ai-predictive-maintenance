"""
JSON API - eight read/write endpoints exposing the platform as a data feed.

    GET  /api/machines            monitored asset register
    GET  /api/fleet-status        live per-machine assessment
    GET  /api/predictions         stored prediction archive
    GET  /api/alerts              stored alerts
    GET  /api/kpis                dashboard KPI block
    GET  /api/kiln-stats          kiln availability analytics
    GET  /api/feature-importance  model card, metrics, collinearity audit
    POST /api/chat                plant assistant

Every endpoint requires an authenticated session.  A real SCADA or IoT
integration would need a token or mTLS scheme instead; session cookies only
serve the browser front end.
"""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

import config
from routes.auth import login_required
from services import (
    chatbot,
    database,
    fleet_service,
    kiln_service,
    ml_service,
)

bp = Blueprint("api", __name__, url_prefix="/api")

FLEET_FIELDS = (
    "machine_id", "machine_name", "machine_type", "department", "category",
    "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min", "thermal_delta_k", "power_w",
    "runtime_hours", "failure_prob", "health_score", "status", "rul_hours",
    "rul_days", "next_service_days", "action", "priority", "priority_label",
)


def _envelope(payload: dict) -> dict:
    payload.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    payload.setdefault("platform", config.BRAND["platform_name"])
    return payload


@bp.get("/machines")
@login_required
def machines():
    return jsonify(_envelope({
        "count": len(config.MACHINES),
        "machines": [
            {
                "id": machine["id"],
                "name": machine["name"],
                "type": machine["type"],
                "type_label": config.MACHINE_TYPES[machine["type"]],
                "department": machine["department"],
                "category": machine["category"],
                "duty_profile": fleet_service.PROFILES.get(machine["id"]),
            }
            for machine in config.MACHINES
        ],
    }))


@bp.get("/fleet-status")
@login_required
def fleet_status():
    fleet = fleet_service.snapshot()
    return jsonify(_envelope({
        "refresh_seconds": fleet_service.REFRESH_SECONDS,
        "summary": fleet_service.summary(fleet),
        "machines": [
            {field: machine.get(field) for field in FLEET_FIELDS}
            | {"drivers": [
                {
                    "label": driver["label"],
                    "severity": driver["severity"],
                    "value": driver["value"],
                }
                for driver in machine["drivers"]
            ]}
            for machine in fleet
        ],
    }))


@bp.get("/predictions")
@login_required
def predictions():
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
    except ValueError:
        limit = 50
    status = request.args.get("status")

    return jsonify(_envelope({
        "stats": database.prediction_stats(),
        "count": limit,
        "records": database.list_predictions(limit=limit, status=status),
    }))


@bp.get("/alerts")
@login_required
def alerts():
    severity = request.args.get("severity")
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
    except ValueError:
        limit = 50

    return jsonify(_envelope({
        "counts": database.alert_counts(),
        "alerts": database.list_alerts(severity=severity, limit=limit),
    }))


@bp.get("/kpis")
@login_required
def kpis():
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    stats = database.prediction_stats()

    return jsonify(_envelope({
        "live": {
            "machines": summary["total"],
            "healthy": summary["healthy"],
            "warning": summary["warning"],
            "critical": summary["critical"],
            "avg_health": summary["avg_health"],
            "avg_failure_prob": summary["avg_failure"],
            "avg_rul_hours": summary["avg_rul_hours"],
            "avg_rul_days": summary["avg_rul_days"],
            "availability": summary["availability"],
            "failure_risk_pct": summary["failure_risk_pct"],
            "priority_counts": summary["priority_counts"],
            "estimated_man_hours": summary["man_hours"],
            "mttr_estimate": summary["mttr_estimate"],
        },
        "archive": stats,
        "trend": fleet_service.health_trend(12),
        "distribution": fleet_service.status_distribution(fleet),
        "daily_volume": database.daily_prediction_volume(7),
    }))


@bp.get("/kiln-stats")
@login_required
def kiln_stats():
    try:
        payload = {
            "kpis": kiln_service.kpis(),
            "causes": kiln_service.cause_breakdown(),
            "sections": kiln_service.section_breakdown(),
            "departments": kiln_service.department_breakdown(),
            "monthly": kiln_service.monthly_trend(),
            "shifts": kiln_service.shift_breakdown(),
            "duration_histogram": kiln_service.duration_histogram(),
            "longest": kiln_service.longest_stoppages(8),
        }
    except kiln_service.KilnDataUnavailable as error:
        return jsonify(_envelope({"error": str(error)})), 503

    return jsonify(_envelope(payload))


@bp.get("/feature-importance")
@login_required
def feature_importance():
    return jsonify(_envelope({
        "model": ml_service.model_card(),
        "metrics": ml_service.metrics(),
        "confusion_matrix": ml_service.confusion(),
        "feature_importance": ml_service.feature_importance(),
        "permutation_importance": ml_service.permutation_importance(),
        "collinearity": ml_service.collinearity(),
        "failure_modes": ml_service.failure_modes(),
    }))


@bp.post("/chat")
@login_required
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", ""))[:500]
    answer = chatbot.reply(message)
    return jsonify(_envelope({
        "message": message,
        "reply": answer["reply"],
        "suggestions": answer["suggestions"],
    }))
