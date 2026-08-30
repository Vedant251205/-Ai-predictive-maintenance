"""
Prediction history: the stored archive of every assessment the platform made.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

from flask import Blueprint, Response, render_template, request

import config
from routes.auth import login_required
from services import database

bp = Blueprint("history", __name__)

EXPORT_COLUMNS = [
    "id", "created_at", "machine_id", "machine_name", "machine_type",
    "department", "air_temperature_k", "process_temperature_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min", "runtime_hours",
    "failure_prob", "health_score", "status", "rul_hours",
    "next_service_days", "action", "priority", "created_by",
]


@bp.route("/history")
@login_required
def records():
    status = request.args.get("status", "all")
    search = request.args.get("q", "").strip()

    rows = database.list_predictions(status=status, search=search or None)

    return render_template(
        "history.html",
        page_title="Prediction History",
        records=rows,
        stats=database.prediction_stats(),
        status=status,
        search=search,
        status_options=["all", "Excellent", "Good", "Warning", "Critical"],
        database_name=config.DATABASE_PATH.name,
    )


@bp.route("/history/export.csv")
@login_required
def export_csv():
    rows = database.list_predictions()

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=EXPORT_COLUMNS, extrasaction="ignore"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="PREDICTIONS_EXPORT_{stamp}.csv"'
            )
        },
    )
