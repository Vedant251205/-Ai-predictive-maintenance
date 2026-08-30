"""Presentation helpers shared by the templates and the JSON API."""

from __future__ import annotations

from datetime import datetime

STATUS_COLOURS = {
    "Excellent": "success",
    "Good": "info",
    "Warning": "warning",
    "Critical": "danger",
}

STATUS_ICONS = {
    "Excellent": "fa-circle-check",
    "Good": "fa-shield-halved",
    "Warning": "fa-triangle-exclamation",
    "Critical": "fa-circle-exclamation",
}


def status_colour(status: str) -> str:
    return STATUS_COLOURS.get(status, "muted")


def status_icon(status: str) -> str:
    return STATUS_ICONS.get(status, "fa-circle")


def number(value, decimals: int = 1) -> str:
    """Format a number with thousands separators, tolerating None."""
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if decimals == 0:
        return f"{numeric:,.0f}"
    return f"{numeric:,.{decimals}f}"


def percent(value, decimals: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{decimals}f}%"


def hours(value, decimals: int = 0) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.{decimals}f} h"


def hours_to_days(value) -> float:
    if value is None:
        return 0.0
    return round(float(value) / 24.0, 1)


def parse_timestamp(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value)[:19], pattern)
        except ValueError:
            continue
    return None


def relative_time(value) -> str:
    """Human friendly age of a timestamp, e.g. '4 min ago'."""
    moment = parse_timestamp(value)
    if moment is None:
        return "-"
    delta = datetime.now() - moment
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return f"{seconds} sec ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours_ago = minutes // 60
    if hours_ago < 24:
        return f"{hours_ago} hr ago"
    days = hours_ago // 24
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    months = days // 30
    return f"{months} mo ago"


def clock_date(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime("%d/%m/%Y")


def clock_time(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime("%H:%M:%S")


def long_date(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime("%A, %b %d, %Y")


def gauge_offset(score: float, radius: float = 54.0) -> float:
    """Stroke-dashoffset for the circular health gauge in the result card."""
    circumference = 2 * 3.141592653589793 * radius
    fraction = max(0.0, min(float(score), 100.0)) / 100.0
    return round(circumference * (1.0 - fraction), 2)


def gauge_circumference(radius: float = 54.0) -> float:
    return round(2 * 3.141592653589793 * radius, 2)


def register(app) -> None:
    """Expose the helpers to Jinja as filters and globals."""
    app.jinja_env.filters["num"] = number
    app.jinja_env.filters["pct"] = percent
    app.jinja_env.filters["hrs"] = hours
    app.jinja_env.filters["to_days"] = hours_to_days
    app.jinja_env.filters["ago"] = relative_time
    app.jinja_env.filters["status_colour"] = status_colour
    app.jinja_env.filters["status_icon"] = status_icon
    app.jinja_env.globals["gauge_offset"] = gauge_offset
    app.jinja_env.globals["gauge_circumference"] = gauge_circumference
    app.jinja_env.globals["long_date"] = long_date
