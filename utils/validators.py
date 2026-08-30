"""Input validation for the predictive cockpit and the admin console."""

from __future__ import annotations

import re

import config

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def validate_sensor_form(form) -> tuple[dict, list[str]]:
    """Coerce and range-check the cockpit form.

    Returns the cleaned payload plus a list of human readable errors.  Values
    are clamped to the physical envelope declared in config.OPERATING_RANGES so
    a malformed slider can never reach the model.
    """
    errors: list[str] = []
    cleaned: dict = {}

    for field, bounds in config.OPERATING_RANGES.items():
        raw = form.get(field, "")
        label = config.FEATURE_LABELS[field]
        if raw is None or str(raw).strip() == "":
            errors.append(f"{label} is required.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if value < bounds["min"] or value > bounds["max"]:
            errors.append(
                f"{label} must be between {bounds['min']:g} and "
                f"{bounds['max']:g} {bounds['unit']}."
            )
            continue
        cleaned[field] = int(round(value)) if isinstance(bounds["step"], int) else value

    machine_type = str(form.get("machine_type", "")).strip().upper()
    if machine_type not in config.MACHINE_TYPES:
        errors.append("Machine type must be L, M or H.")
    else:
        cleaned["machine_type"] = machine_type

    machine_id = str(form.get("machine_id", "")).strip().upper()
    if machine_id and machine_id not in config.MACHINE_INDEX:
        errors.append(f"Unknown machine id '{machine_id}'.")
    else:
        cleaned["machine_id"] = machine_id

    raw_runtime = form.get("runtime_hours", "") or "0"
    try:
        runtime = float(raw_runtime)
        if runtime < 0 or runtime > 200000:
            errors.append("Total runtime must be between 0 and 200000 hours.")
        else:
            cleaned["runtime_hours"] = round(runtime, 1)
    except (TypeError, ValueError):
        errors.append("Total runtime must be a number.")

    return cleaned, errors


def validate_employee_form(form, existing_ids: set[str]) -> tuple[dict, list[str]]:
    """Validate the admin 'create employee account' form."""
    errors: list[str] = []

    user_id = str(form.get("user_id", "")).strip()
    name = str(form.get("name", "")).strip()
    email = str(form.get("email", "")).strip().lower()
    password = str(form.get("password", ""))
    role = str(form.get("role", "employee")).strip().lower()
    department = str(form.get("department", "")).strip() or "Operations"

    if not USER_ID_PATTERN.match(user_id):
        errors.append(
            "Employee ID must be 3-32 characters using letters, digits, "
            "dot, dash or underscore."
        )
    elif user_id.lower() in {value.lower() for value in existing_ids}:
        errors.append(f"Employee ID '{user_id}' already exists.")

    if len(name) < 3 or len(name) > 64:
        errors.append("Full name must be between 3 and 64 characters.")

    if not EMAIL_PATTERN.match(email):
        errors.append("Enter a valid corporate email address.")

    if len(password) < 6:
        errors.append("Access password must be at least 6 characters.")

    valid_roles = {value for value, _ in config.ROLE_OPTIONS}
    if role not in valid_roles:
        errors.append("Operational role is not recognised.")

    payload = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "password": password,
        "role": role,
        "department": department,
    }
    return payload, errors


def validate_alert_settings(form) -> tuple[dict, list[str]]:
    """Validate the Alert Center notification gateway form."""
    errors: list[str] = []

    email_enabled = 1 if form.get("email_enabled") in ("on", "1", "true") else 0
    sms_enabled = 1 if form.get("sms_enabled") in ("on", "1", "true") else 0
    recipient = str(form.get("recipient_email", "")).strip().lower()
    phone = str(form.get("recipient_phone", "")).strip()
    severity = str(form.get("severity", "critical")).strip().lower()

    if recipient and not EMAIL_PATTERN.match(recipient):
        errors.append("Recipient email address is not valid.")
    if email_enabled and not recipient:
        errors.append("Enter a recipient email address before enabling email alerts.")

    if phone and not re.match(r"^\+?[0-9\s-]{8,18}$", phone):
        errors.append("Recipient phone number is not valid.")
    if sms_enabled and not phone:
        errors.append("Enter a recipient phone number before enabling SMS alerts.")

    valid_severities = {value for value, _ in config.SEVERITY_OPTIONS}
    if severity not in valid_severities:
        errors.append("Alert severity threshold is not recognised.")

    payload = {
        "email_enabled": email_enabled,
        "sms_enabled": sms_enabled,
        "recipient_email": recipient,
        "recipient_phone": phone,
        "severity": severity,
    }
    return payload, errors
