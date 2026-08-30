"""
SQLite persistence layer  (instance/predictive_maintenance.db).

Tables
------
users            role-based accounts managed from the admin console
predictions      every AI prediction ever generated
alerts           warning / critical events raised by the alert engine
alert_settings   single-row notification gateway configuration
audit_logs       who did what, when

All statements are parameterised.  Passwords are stored as Werkzeug PBKDF2
hashes, never in clear text.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Iterator

from werkzeug.security import check_password_hash, generate_password_hash

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'employee',
    department    TEXT    NOT NULL DEFAULT 'Operations',
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL,
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id            TEXT,
    machine_name          TEXT,
    machine_type          TEXT    NOT NULL,
    department            TEXT,
    air_temperature_k     REAL    NOT NULL,
    process_temperature_k REAL    NOT NULL,
    rotational_speed_rpm  INTEGER NOT NULL,
    torque_nm             REAL    NOT NULL,
    tool_wear_min         INTEGER NOT NULL,
    runtime_hours         REAL    NOT NULL DEFAULT 0,
    failure_prob          REAL    NOT NULL,
    health_score          REAL    NOT NULL,
    status                TEXT    NOT NULL,
    rul_hours             REAL    NOT NULL,
    next_service_days     INTEGER NOT NULL,
    action                TEXT    NOT NULL,
    priority              TEXT    NOT NULL,
    created_by            TEXT,
    created_at            TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id   TEXT    NOT NULL,
    severity     TEXT    NOT NULL,
    title        TEXT    NOT NULL,
    message      TEXT    NOT NULL,
    channel      TEXT    NOT NULL DEFAULT 'system',
    acknowledged INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL,
    acknowledged_at TEXT
);

CREATE TABLE IF NOT EXISTS alert_settings (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    email_enabled   INTEGER NOT NULL DEFAULT 0,
    sms_enabled     INTEGER NOT NULL DEFAULT 0,
    recipient_email TEXT    NOT NULL DEFAULT '',
    recipient_phone TEXT    NOT NULL DEFAULT '',
    severity        TEXT    NOT NULL DEFAULT 'critical',
    updated_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_created
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_machine
    ON predictions (machine_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created
    ON alerts (created_at DESC);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def init_db() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)

        # Seed accounts are inserted when missing and re-synced when present.
        #
        # Re-syncing matters for security: the seeded passwords come from the
        # environment, and on a host that keeps its disk between deploys an
        # insert-only seed would silently leave the previous password working
        # after an operator had changed the environment variable.
        for seed in config.SEED_USERS:
            row = connection.execute(
                "SELECT id FROM users WHERE user_id = ?", (seed["user_id"],)
            ).fetchone()

            if row is None:
                connection.execute(
                    """INSERT INTO users
                       (user_id, name, email, password_hash, role, department,
                        active, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                    (
                        seed["user_id"],
                        seed["name"],
                        seed["email"].lower(),
                        generate_password_hash(seed["password"]),
                        seed["role"],
                        seed["department"],
                        _now(),
                    ),
                )
            else:
                connection.execute(
                    """UPDATE users
                       SET name = ?, email = ?, password_hash = ?, role = ?,
                           department = ?, active = 1
                       WHERE user_id = ?""",
                    (
                        seed["name"],
                        seed["email"].lower(),
                        generate_password_hash(seed["password"]),
                        seed["role"],
                        seed["department"],
                        seed["user_id"],
                    ),
                )

        settings = connection.execute(
            "SELECT COUNT(*) AS n FROM alert_settings"
        ).fetchone()["n"]
        if settings == 0:
            connection.execute(
                """INSERT INTO alert_settings
                   (id, email_enabled, sms_enabled, recipient_email,
                    recipient_phone, severity, updated_at)
                   VALUES (1, 0, 0, '', '', 'critical', ?)""",
                (_now(),),
            )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def authenticate(identifier: str, user_id: str, password: str,
                 role: str) -> dict | None:
    """Verify credentials.  `identifier` is the email, `user_id` the staff id."""
    with connect() as connection:
        row = connection.execute(
            """SELECT * FROM users
               WHERE lower(email) = lower(?) AND lower(user_id) = lower(?)
                 AND role = ? AND active = 1""",
            (identifier.strip(), user_id.strip(), role),
        ).fetchone()

        if row is None or not check_password_hash(row["password_hash"], password):
            return None

        connection.execute(
            "UPDATE users SET last_login = ? WHERE id = ?", (_now(), row["id"])
        )
        return dict(row)


def list_users(include_inactive: bool = True) -> list[dict]:
    query = "SELECT * FROM users"
    if not include_inactive:
        query += " WHERE active = 1"
    query += " ORDER BY role DESC, user_id ASC"
    with connect() as connection:
        return rows_to_dicts(connection.execute(query).fetchall())


def get_user_ids() -> set[str]:
    with connect() as connection:
        rows = connection.execute("SELECT user_id FROM users").fetchall()
    return {row["user_id"] for row in rows}


def email_exists(email: str) -> bool:
    with connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM users WHERE lower(email) = lower(?)", (email,)
        ).fetchone()
    return row is not None


def create_user(payload: dict) -> None:
    with connect() as connection:
        connection.execute(
            """INSERT INTO users
               (user_id, name, email, password_hash, role, department,
                active, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                payload["user_id"],
                payload["name"],
                payload["email"].lower(),
                generate_password_hash(payload["password"]),
                payload["role"],
                payload["department"],
                _now(),
            ),
        )


def revoke_user(user_id: str) -> bool:
    """Deactivate an account.  Returns False when the target is protected."""
    with connect() as connection:
        row = connection.execute(
            "SELECT role FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return False
        if row["role"] == "admin":
            remaining = connection.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1"
            ).fetchone()["n"]
            if remaining <= 1:
                return False
        connection.execute(
            "UPDATE users SET active = 0 WHERE user_id = ?", (user_id,)
        )
        return True


def restore_user(user_id: str) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE users SET active = 1 WHERE user_id = ?", (user_id,)
        )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
PREDICTION_COLUMNS = (
    "machine_id", "machine_name", "machine_type", "department",
    "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min", "runtime_hours", "failure_prob",
    "health_score", "status", "rul_hours", "next_service_days", "action",
    "priority", "created_by",
)


def save_prediction(prediction: dict, actor: str) -> int:
    values = [prediction.get(column) for column in PREDICTION_COLUMNS[:-1]]
    values.append(actor)
    placeholders = ", ".join("?" for _ in PREDICTION_COLUMNS)
    columns = ", ".join(PREDICTION_COLUMNS)
    with connect() as connection:
        cursor = connection.execute(
            f"INSERT INTO predictions ({columns}, created_at) "
            f"VALUES ({placeholders}, ?)",
            (*values, _now()),
        )
        return int(cursor.lastrowid)


def list_predictions(limit: int | None = None, status: str | None = None,
                     machine_id: str | None = None,
                     search: str | None = None) -> list[dict]:
    clauses: list[str] = []
    params: list = []

    if status and status.lower() != "all":
        clauses.append("status = ?")
        params.append(status)
    if machine_id:
        clauses.append("machine_id = ?")
        params.append(machine_id)
    if search:
        clauses.append(
            "(machine_id LIKE ? OR machine_name LIKE ? OR status LIKE ? "
            "OR priority LIKE ?)"
        )
        pattern = f"%{search}%"
        params.extend([pattern] * 4)

    query = "SELECT * FROM predictions"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))

    with connect() as connection:
        return rows_to_dicts(connection.execute(query, params).fetchall())


def prediction_stats() -> dict:
    with connect() as connection:
        row = connection.execute(
            """SELECT
                   COUNT(*)                                        AS total,
                   COALESCE(AVG(health_score), 0)                   AS avg_health,
                   COALESCE(AVG(failure_prob), 0)                   AS avg_failure,
                   COALESCE(AVG(rul_hours), 0)                      AS avg_rul,
                   COALESCE(SUM(status = 'Excellent'), 0)           AS excellent,
                   COALESCE(SUM(status = 'Good'), 0)                AS good,
                   COALESCE(SUM(status = 'Warning'), 0)             AS warning,
                   COALESCE(SUM(status = 'Critical'), 0)            AS critical
               FROM predictions"""
        ).fetchone()
    stats = dict(row)
    stats["healthy"] = stats["excellent"] + stats["good"]
    stats["non_healthy"] = stats["warning"] + stats["critical"]
    stats["avg_health"] = round(float(stats["avg_health"]), 1)
    stats["avg_failure"] = round(float(stats["avg_failure"]), 1)
    stats["avg_rul"] = round(float(stats["avg_rul"]), 0)
    return stats


def daily_prediction_volume(days: int = 7) -> list[dict]:
    """Total runs vs flagged risks per day, for the alert-volume trend chart."""
    with connect() as connection:
        rows = connection.execute(
            """SELECT substr(created_at, 1, 10) AS day,
                      COUNT(*)                  AS total,
                      SUM(status IN ('Warning', 'Critical')) AS risks
               FROM predictions
               GROUP BY day
               ORDER BY day DESC
               LIMIT ?""",
            (int(days),),
        ).fetchall()
    return list(reversed(rows_to_dicts(rows)))


def machine_latest_predictions() -> dict[str, dict]:
    """Most recent stored prediction per machine id."""
    with connect() as connection:
        rows = connection.execute(
            """SELECT p.* FROM predictions p
               JOIN (SELECT machine_id, MAX(id) AS newest
                     FROM predictions
                     WHERE machine_id IS NOT NULL AND machine_id <> ''
                     GROUP BY machine_id) latest
                 ON p.machine_id = latest.machine_id AND p.id = latest.newest"""
        ).fetchall()
    return {row["machine_id"]: dict(row) for row in rows}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def add_alert(machine_id: str, severity: str, title: str, message: str,
              channel: str = "system") -> int:
    with connect() as connection:
        cursor = connection.execute(
            """INSERT INTO alerts
               (machine_id, severity, title, message, channel, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (machine_id, severity, title, message, channel, _now()),
        )
        return int(cursor.lastrowid)


def list_alerts(severity: str | None = None,
                limit: int | None = None) -> list[dict]:
    query = "SELECT * FROM alerts"
    params: list = []
    if severity:
        query += " WHERE severity = ?"
        params.append(severity)
    query += " ORDER BY id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))
    with connect() as connection:
        return rows_to_dicts(connection.execute(query, params).fetchall())


def alert_counts() -> dict:
    with connect() as connection:
        row = connection.execute(
            """SELECT
                   COUNT(*)                                   AS total,
                   COALESCE(SUM(severity = 'critical'), 0)    AS critical,
                   COALESCE(SUM(severity = 'warning'), 0)     AS warning,
                   COALESCE(SUM(acknowledged = 0), 0)         AS unacknowledged
               FROM alerts"""
        ).fetchone()
    return dict(row)


def acknowledge_alerts(severity: str | None = None,
                       alert_id: int | None = None) -> int:
    query = "UPDATE alerts SET acknowledged = 1, acknowledged_at = ? WHERE acknowledged = 0"
    params: list = [_now()]
    if alert_id is not None:
        query += " AND id = ?"
        params.append(int(alert_id))
    elif severity:
        query += " AND severity = ?"
        params.append(severity)
    with connect() as connection:
        cursor = connection.execute(query, params)
        return int(cursor.rowcount)


# ---------------------------------------------------------------------------
# Alert settings
# ---------------------------------------------------------------------------
def get_alert_settings() -> dict:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM alert_settings WHERE id = 1"
        ).fetchone()
    if row is None:
        return {
            "email_enabled": 0, "sms_enabled": 0, "recipient_email": "",
            "recipient_phone": "", "severity": "critical", "updated_at": _now(),
        }
    return dict(row)


def save_alert_settings(payload: dict) -> None:
    with connect() as connection:
        connection.execute(
            """UPDATE alert_settings
               SET email_enabled = ?, sms_enabled = ?, recipient_email = ?,
                   recipient_phone = ?, severity = ?, updated_at = ?
               WHERE id = 1""",
            (
                payload["email_enabled"],
                payload["sms_enabled"],
                payload["recipient_email"],
                payload["recipient_phone"],
                payload["severity"],
                _now(),
            ),
        )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
def log_action(actor: str, action: str, detail: str = "") -> None:
    with connect() as connection:
        connection.execute(
            """INSERT INTO audit_logs (actor, action, detail, created_at)
               VALUES (?, ?, ?, ?)""",
            (actor, action, detail, _now()),
        )


def list_audit_logs(limit: int = 50) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return rows_to_dicts(rows)


def audit_count() -> int:
    with connect() as connection:
        row = connection.execute("SELECT COUNT(*) AS n FROM audit_logs").fetchone()
    return int(row["n"])


def recent_alert_exists(machine_id: str, severity: str,
                        minutes: int = 15) -> bool:
    """True when an equivalent alert was already raised very recently.

    Stops the alert engine from re-raising the same condition on every page
    render while the underlying fault is still present.
    """
    with connect() as connection:
        row = connection.execute(
            """SELECT 1 FROM alerts
               WHERE machine_id = ? AND severity = ?
                 AND created_at >= datetime('now', 'localtime', ?)
               LIMIT 1""",
            (machine_id, severity, f"-{int(minutes)} minutes"),
        ).fetchone()
    return row is not None
