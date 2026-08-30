"""
Primary page routes: dashboards, alert center and the analytics modules.
"""

from __future__ import annotations

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

import config
from routes.auth import actor_label, admin_required, login_required
from services import (
    alert_service,
    database,
    fleet_service,
    kiln_service,
    ml_service,
)
from utils import validators

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return redirect(url_for("main.dashboard"))


# ---------------------------------------------------------------------------
# Dashboards
# ---------------------------------------------------------------------------
@bp.route("/dashboard")
@login_required
def dashboard():
    from flask import session

    if session.get("role") == "admin":
        return redirect(url_for("main.admin_dashboard"))
    return redirect(url_for("main.employee_dashboard"))


@bp.route("/dashboard/admin")
@admin_required
def admin_dashboard():
    users = database.list_users()
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    stats = database.prediction_stats()

    return render_template(
        "dashboard_admin.html",
        page_title="Admin Control Panel",
        users=users,
        fleet=fleet,
        summary=summary,
        stats=stats,
        role_options=config.ROLE_OPTIONS,
        departments=config.DEPARTMENTS + ["Management", "Operations", "Maintenance"],
        audit_logs=database.list_audit_logs(12),
    )


@bp.route("/dashboard/employee")
@login_required
def employee_dashboard():
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    stats = database.prediction_stats()

    return render_template(
        "dashboard_employee.html",
        page_title="Operations Dashboard",
        fleet=fleet,
        summary=summary,
        stats=stats,
        health_series=fleet_service.machine_health_series(fleet),
        distribution=fleet_service.status_distribution(fleet),
        recent=database.list_predictions(limit=8),
        alerts=database.list_alerts(limit=6),
    )


# ---------------------------------------------------------------------------
# Alert Center
# ---------------------------------------------------------------------------
@bp.route("/alerts")
@login_required
def alerts():
    fleet = fleet_service.snapshot()
    return render_template(
        "alerts.html",
        page_title="Alert Center Console",
        fleet=fleet,
        summary=alert_service.centre_summary(fleet),
        gateways=alert_service.gateway_status(),
        critical_alerts=database.list_alerts(severity="critical", limit=25),
        warning_alerts=database.list_alerts(severity="warning", limit=25),
        audit_logs=database.list_audit_logs(25),
        severity_options=config.SEVERITY_OPTIONS,
        thresholds=config.ALERT_THRESHOLDS,
    )


@bp.route("/alerts/scan", methods=["POST"])
@login_required
def scan_alerts():
    fleet = fleet_service.snapshot()
    result = alert_service.scan_fleet(fleet, actor_label())

    if result["created"]:
        flash(
            f"Fleet scan raised {result['created']} new alert(s).",
            "success",
        )
    else:
        flash(
            "Fleet scan complete. No new conditions to report "
            f"({result['skipped']} already open).",
            "info",
        )
    for message in result["dispatched"]:
        flash(message, "info")

    database.log_action(
        actor_label(), "fleet_scan",
        f"{result['created']} created, {result['skipped']} deduplicated.",
    )
    return redirect(url_for("main.alerts"))


@bp.route("/alerts/acknowledge", methods=["POST"])
@login_required
def acknowledge_alerts():
    severity = request.form.get("severity") or None
    alert_id = request.form.get("alert_id")
    count = database.acknowledge_alerts(
        severity=severity,
        alert_id=int(alert_id) if alert_id and alert_id.isdigit() else None,
    )
    database.log_action(
        actor_label(), "alerts_acknowledged",
        f"{count} alert(s) acknowledged ({severity or 'single'}).",
    )
    flash(f"{count} alert(s) acknowledged.", "success")
    return redirect(url_for("main.alerts"))


@bp.route("/alerts/settings", methods=["POST"])
@login_required
def save_alert_settings():
    payload, errors = validators.validate_alert_settings(request.form)
    if errors:
        for message in errors:
            flash(message, "danger")
        return redirect(url_for("main.alerts"))

    database.save_alert_settings(payload)
    database.log_action(
        actor_label(), "alert_settings_saved",
        f"email={bool(payload['email_enabled'])} "
        f"sms={bool(payload['sms_enabled'])} severity={payload['severity']}",
    )

    flash("Notification gateway settings saved.", "success")
    if payload["email_enabled"] and not alert_service.email_ready():
        flash(
            "Email is switched on but no SMTP credentials are present. Add "
            "PLATFORM_EMAIL and PLATFORM_EMAIL_PASSWORD to .env and set "
            "EMAIL_ENABLED=1, then restart.",
            "warning",
        )
    if payload["sms_enabled"] and not alert_service.sms_ready():
        flash(
            "SMS is switched on but the Twilio gateway is not available. Add "
            "your Twilio credentials to .env, set SMS_ENABLED=1, install the "
            "twilio package, then restart.",
            "warning",
        )
    return redirect(url_for("main.alerts"))


@bp.route("/alerts/test", methods=["POST"])
@login_required
def test_alert():
    for message in alert_service.send_test_alert(actor_label()):
        flash(message, "info")
    return redirect(url_for("main.alerts"))


# ---------------------------------------------------------------------------
# Executive dashboard
# ---------------------------------------------------------------------------
@bp.route("/executive-dashboard")
@login_required
def executive_dashboard():
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    stats = database.prediction_stats()

    # The executive view reports on the whole prediction archive, falling back
    # to live fleet figures until the archive has records.
    if stats["total"]:
        archive_health = stats["avg_health"]
        archive_failure = stats["avg_failure"]
        archive_rul = stats["avg_rul"]
        availability = round(
            stats["healthy"] / stats["total"] * 100.0, 1
        )
        failure_risk = round(
            stats["non_healthy"] / stats["total"] * 100.0, 1
        )
    else:
        archive_health = summary["avg_health"]
        archive_failure = summary["avg_failure"]
        archive_rul = summary["avg_rul_hours"]
        availability = summary["availability"]
        failure_risk = summary["failure_risk_pct"]

    return render_template(
        "executive_dashboard.html",
        page_title="Executive Dashboard",
        fleet=fleet,
        summary=summary,
        stats=stats,
        archive={
            "avg_health": archive_health,
            "avg_failure": archive_failure,
            "avg_rul": archive_rul,
            "availability": availability,
            "failure_risk": failure_risk,
        },
        health_series=fleet_service.machine_health_series(fleet),
        trend=fleet_service.health_trend(12),
        distribution=fleet_service.status_distribution(fleet),
        kiln=_safe_kiln_kpis(),
        metrics=ml_service.metrics(),
    )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
@bp.route("/analytics")
@login_required
def analytics():
    fleet = fleet_service.snapshot()
    stats = database.prediction_stats()

    return render_template(
        "analytics.html",
        page_title="Analytics Center",
        fleet=fleet,
        summary=fleet_service.summary(fleet),
        stats=stats,
        distribution=fleet_service.status_distribution(fleet),
        health_series=fleet_service.machine_health_series(fleet),
        volume=database.daily_prediction_volume(7),
        metrics=ml_service.metrics(),
        importance=ml_service.feature_importance(),
        recent=database.list_predictions(limit=10),
        database_name=config.DATABASE_PATH.name,
    )


@bp.route("/feature-intelligence")
@login_required
def feature_intelligence():
    return render_template(
        "feature_intelligence.html",
        page_title="AI Feature Intelligence",
        card=ml_service.model_card(),
        metrics=ml_service.metrics(),
        confusion=ml_service.confusion(),
        importance=ml_service.feature_importance(),
        permutation=ml_service.permutation_importance(),
        collinearity=ml_service.collinearity(),
        failure_modes=ml_service.failure_modes(),
        ranges=config.OPERATING_RANGES,
        labels=config.FEATURE_LABELS,
    )


@bp.route("/kiln-analytics")
@login_required
def kiln_analytics():
    try:
        data = kiln_service.kpis()
    except kiln_service.KilnDataUnavailable as error:
        flash(str(error), "danger")
        return render_template(
            "kiln_analytics.html",
            page_title="Kiln Stoppage Analytics",
            unavailable=True,
        )

    return render_template(
        "kiln_analytics.html",
        page_title="Kiln Stoppage Analytics",
        unavailable=False,
        kpis=data,
        causes=kiln_service.cause_breakdown(),
        sections=kiln_service.section_breakdown(),
        departments=kiln_service.department_breakdown(),
        monthly=kiln_service.monthly_trend(),
        shifts=kiln_service.shift_breakdown(),
        histogram=kiln_service.duration_histogram(),
        longest=kiln_service.longest_stoppages(8),
        recent=kiln_service.recent_stoppages(12),
    )


# ---------------------------------------------------------------------------
# Maintenance advisor
# ---------------------------------------------------------------------------
@bp.route("/maintenance-advisor")
@login_required
def maintenance_advisor():
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    orders = fleet_service.work_orders(fleet)

    return render_template(
        "maintenance_advisor.html",
        page_title="Maintenance Advisor",
        fleet=fleet,
        orders=orders,
        summary=summary,
        critical=fleet_service.critical_machines(fleet),
        priorities=config.PRIORITIES,
        departments=config.DEPARTMENTS,
    )


# ---------------------------------------------------------------------------
# Static documentation pages
# ---------------------------------------------------------------------------
@bp.route("/architecture")
@login_required
def architecture():
    return render_template(
        "architecture.html",
        page_title="System Architecture",
        card=ml_service.model_card(),
        metrics=ml_service.metrics(),
        importance=ml_service.feature_importance(),
        tech_stack=config.TECH_STACK,
        machine_count=len(config.MACHINES),
        database_name=config.DATABASE_PATH.name,
    )


@bp.route("/future-roadmap")
@login_required
def future_roadmap():
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    counts = database.alert_counts()
    gateways = alert_service.gateway_status()

    low_rul = sum(1 for item in fleet if item["rul_hours"] < 3600)

    return render_template(
        "future_roadmap.html",
        page_title="Future Roadmap",
        fleet=fleet,
        summary=summary,
        low_rul=low_rul,
        alert_total=counts["total"],
        gateways=gateways,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_kiln_kpis() -> dict | None:
    try:
        return kiln_service.kpis()
    except kiln_service.KilnDataUnavailable:
        return None
