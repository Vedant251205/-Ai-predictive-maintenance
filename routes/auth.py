"""
Authentication and account administration.

Role-based access: an 'admin' session reaches the System Control & Access
Center, an 'employee' session does not.  Credentials are checked against the
PBKDF2 hashes in SQLite; the session stores only identity, never the password.
"""

from __future__ import annotations

from functools import wraps

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

import config
from services import database
from utils import validators

bp = Blueprint("auth", __name__)


# ---------------------------------------------------------------------------
# Access control decorators
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)

    return wrapper


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        if session.get("role") != "admin":
            flash(
                "Administrator clearance is required for that area.", "danger"
            )
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapper


def current_user() -> dict:
    return {
        "user_id": session.get("user_id"),
        "name": session.get("name"),
        "email": session.get("email"),
        "role": session.get("role"),
        "department": session.get("department"),
    }


def actor_label() -> str:
    return session.get("user_id") or "anonymous"


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------
@bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id") and request.method == "GET":
        return redirect(url_for("main.dashboard"))

    role = request.values.get("role", "admin")
    if role not in {"admin", "employee"}:
        role = "admin"

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        user_id = request.form.get("user_id", "").strip()
        password = request.form.get("password", "")

        if not (email and user_id and password):
            flash("All three credentials are required.", "danger")
        else:
            user = database.authenticate(email, user_id, password, role)
            if user is None:
                database.log_action(
                    user_id or email, "login_failed",
                    f"Rejected {role} sign-in attempt.",
                )
                flash(
                    "Authentication failed. Check the email, ID, password and "
                    "the selected clearance level.",
                    "danger",
                )
            else:
                session.clear()
                session["user_id"] = user["user_id"]
                session["name"] = user["name"]
                session["email"] = user["email"]
                session["role"] = user["role"]
                session["department"] = user["department"]
                session.permanent = True

                database.log_action(
                    user["user_id"], "login", f"Signed in as {user['role']}."
                )
                flash(
                    f"Welcome back, {user['name']}! You are logged in as "
                    f"{user['role'].title()}.",
                    "success",
                )
                destination = request.args.get("next")
                if destination and destination.startswith("/"):
                    return redirect(destination)
                return redirect(url_for("main.dashboard"))

    return render_template(
        "login.html",
        role=role,
        demo_keys=config.DEMO_KEYS if config.SHOW_DEMO_KEYS else {},
        show_demo_keys=config.SHOW_DEMO_KEYS,
        hide_shell=True,
    )


@bp.route("/logout", methods=["GET", "POST"])
def logout():
    actor = session.get("user_id")
    if actor:
        database.log_action(actor, "logout", "Session ended.")
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))


# ---------------------------------------------------------------------------
# Account administration  (admin only)
# ---------------------------------------------------------------------------
@bp.route("/admin/users/create", methods=["POST"])
@admin_required
def create_user():
    payload, errors = validators.validate_employee_form(
        request.form, database.get_user_ids()
    )

    if not errors and database.email_exists(payload["email"]):
        errors.append(f"Email '{payload['email']}' is already registered.")

    if errors:
        for message in errors:
            flash(message, "danger")
        return redirect(url_for("main.admin_dashboard"))

    database.create_user(payload)
    database.log_action(
        actor_label(),
        "user_created",
        f"Created {payload['role']} account {payload['user_id']} "
        f"({payload['department']}).",
    )
    flash(
        f"Account {payload['user_id']} created for {payload['name']}.",
        "success",
    )
    return redirect(url_for("main.admin_dashboard"))


@bp.route("/admin/users/<user_id>/revoke", methods=["POST"])
@admin_required
def revoke_user(user_id: str):
    if user_id == session.get("user_id"):
        flash("You cannot revoke the account you are signed in with.", "danger")
        return redirect(url_for("main.admin_dashboard"))

    if database.revoke_user(user_id):
        database.log_action(
            actor_label(), "user_revoked", f"Revoked access for {user_id}."
        )
        flash(f"Access revoked for {user_id}.", "success")
    else:
        flash(
            f"Could not revoke {user_id}. The last active administrator "
            "cannot be removed.",
            "danger",
        )
    return redirect(url_for("main.admin_dashboard"))


@bp.route("/admin/users/<user_id>/restore", methods=["POST"])
@admin_required
def restore_user(user_id: str):
    if user_id not in database.get_user_ids():
        abort(404)
    database.restore_user(user_id)
    database.log_action(
        actor_label(), "user_restored", f"Restored access for {user_id}."
    )
    flash(f"Access restored for {user_id}.", "success")
    return redirect(url_for("main.admin_dashboard"))
