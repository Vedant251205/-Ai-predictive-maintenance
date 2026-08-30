"""
Alert engine and notification gateways.

Threshold breaches raised by the fleet are persisted as alerts and, when the
administrator has configured a gateway, dispatched over email (SMTP) or SMS
(Twilio).  Both gateways fail soft: if credentials are absent or the provider
rejects the request, the attempt is recorded in the audit trail and the platform
keeps running.  Nothing here ever blocks a page render on a network call it
cannot complete.
"""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

import config
from services import database

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


# ---------------------------------------------------------------------------
# Gateway readiness
# ---------------------------------------------------------------------------
def email_ready() -> bool:
    return bool(
        config.EMAIL["enabled"]
        and config.EMAIL["sender"]
        and config.EMAIL["password"]
    )


def sms_ready() -> bool:
    if not config.SMS["enabled"]:
        return False
    if not (config.SMS["account_sid"] and config.SMS["auth_token"]
            and config.SMS["from_number"]):
        return False
    try:
        import twilio  # noqa: F401
    except ImportError:
        return False
    return True


def gateway_status() -> dict:
    settings = database.get_alert_settings()
    email_configured = email_ready()
    sms_configured = sms_ready()

    if email_configured and settings["email_enabled"]:
        email_state, email_label = "success", "READY"
    elif settings["email_enabled"]:
        email_state, email_label = "warning", "NOT CONFIGURED"
    else:
        email_state, email_label = "muted", "DISABLED"

    if sms_configured and settings["sms_enabled"]:
        sms_state, sms_label = "success", "READY"
    elif settings["sms_enabled"]:
        sms_state, sms_label = "warning", "NOT CONFIGURED"
    else:
        sms_state, sms_label = "muted", "DISABLED"

    return {
        "settings": settings,
        "email": {
            "state": email_state,
            "label": email_label,
            "configured": email_configured,
            "sender": config.EMAIL["sender"] or "-",
            "host": config.EMAIL["host"],
            "port": config.EMAIL["port"],
        },
        "sms": {
            "state": sms_state,
            "label": sms_label,
            "configured": sms_configured,
            "from_number": config.SMS["from_number"] or "-",
        },
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _send_email(recipient: str, subject: str, body: str) -> tuple[bool, str]:
    if not email_ready():
        return False, "Email gateway is not configured."
    if not recipient:
        return False, "No recipient email address configured."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = (
        f"{config.EMAIL['display_name']} <{config.EMAIL['sender']}>"
    )
    message["To"] = recipient
    message.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(config.EMAIL["host"], config.EMAIL["port"],
                          timeout=12) as server:
            server.starttls(context=context)
            server.login(config.EMAIL["sender"], config.EMAIL["password"])
            server.send_message(message)
        return True, f"Email dispatched to {recipient}."
    except Exception as error:                      # noqa: BLE001
        return False, f"Email dispatch failed: {error}"


def _send_sms(recipient: str, body: str) -> tuple[bool, str]:
    if not sms_ready():
        return False, "SMS gateway is not configured."
    if not recipient:
        return False, "No recipient phone number configured."

    try:
        from twilio.rest import Client

        client = Client(config.SMS["account_sid"], config.SMS["auth_token"])
        client.messages.create(
            body=body[:1500],
            from_=config.SMS["from_number"],
            to=recipient,
        )
        return True, f"SMS dispatched to {recipient}."
    except Exception as error:                      # noqa: BLE001
        return False, f"SMS dispatch failed: {error}"


def dispatch(severity: str, title: str, body: str,
             actor: str = "system") -> list[str]:
    """Push an alert through whichever gateways are enabled and eligible."""
    settings = database.get_alert_settings()
    threshold = settings.get("severity", "critical")

    minimum = {"critical": 2, "warning": 1, "all": 0}.get(threshold, 2)
    if SEVERITY_RANK.get(severity, 0) < minimum:
        return [f"Suppressed: severity '{severity}' is below the "
                f"'{threshold}' dispatch threshold."]

    outcomes: list[str] = []

    if settings.get("email_enabled"):
        success, detail = _send_email(
            settings.get("recipient_email", ""), title, body
        )
        outcomes.append(detail)
        database.log_action(
            actor, "email_alert" if success else "email_alert_failed", detail
        )
    if settings.get("sms_enabled"):
        success, detail = _send_sms(
            settings.get("recipient_phone", ""), f"{title}\n{body}"
        )
        outcomes.append(detail)
        database.log_action(
            actor, "sms_alert" if success else "sms_alert_failed", detail
        )

    if not outcomes:
        outcomes.append(
            "No gateway enabled - alert recorded in the platform only."
        )
    return outcomes


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------
def _describe(machine: dict) -> tuple[str, str]:
    reasons = [item["message"] for item in machine["drivers"][:2]]
    if not reasons:
        reasons = [
            f"Model failure probability {machine['failure_prob']:.1f}% "
            f"with health score {machine['health_score']:.1f}."
        ]
    title = (
        f"{machine['machine_id']} {machine['status'].upper()} - "
        f"health {machine['health_score']:.1f}"
    )
    body = (
        f"{machine['machine_name']} ({machine['department']}) is reporting "
        f"{machine['status'].lower()} condition. "
        f"Failure probability {machine['failure_prob']:.1f}%, remaining useful "
        f"life {machine['rul_hours']:,} h, work-order priority "
        f"{machine['priority']} ({machine['priority_label']}). "
        + " ".join(reasons)
    )
    return title, body


def scan_fleet(fleet: list[dict], actor: str = "system",
               dedupe_minutes: int = 15) -> dict:
    """Raise alerts for every machine currently outside its healthy band."""
    created = 0
    skipped = 0
    dispatched: list[str] = []

    for machine in fleet:
        if machine["status"] == "Critical":
            severity = "critical"
        elif machine["status"] == "Warning":
            severity = "warning"
        else:
            continue

        if database.recent_alert_exists(
            machine["machine_id"], severity, dedupe_minutes
        ):
            skipped += 1
            continue

        title, body = _describe(machine)
        database.add_alert(machine["machine_id"], severity, title, body)
        created += 1

        if severity == "critical":
            dispatched.extend(dispatch(severity, title, body, actor))

    return {"created": created, "skipped": skipped, "dispatched": dispatched}


def send_test_alert(actor: str) -> list[str]:
    title = f"{config.BRAND['platform_name']} - test notification"
    body = (
        "This is a test alert raised from the Alert Center. If you received "
        "this message the notification gateway is working."
    )
    database.add_alert("SYSTEM", "info", title, body, channel="test")
    database.log_action(actor, "test_alert", "Test notification triggered.")
    return dispatch("critical", title, body, actor)


def centre_summary(fleet: list[dict]) -> dict:
    """Headline counters for the Alert Center cards."""
    counts = database.alert_counts()
    live_critical = sum(1 for item in fleet if item["status"] == "Critical")
    live_warning = sum(1 for item in fleet if item["status"] == "Warning")
    return {
        "stored_total": counts["total"],
        "stored_critical": counts["critical"],
        "stored_warning": counts["warning"],
        "unacknowledged": counts["unacknowledged"],
        "live_critical": live_critical,
        "live_warning": live_warning,
        "audit_count": database.audit_count(),
    }
