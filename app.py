"""
AI-Based Predictive Maintenance System - application factory.

Run locally:

    python app.py

On first start the factory generates the synthetic datasets, trains the Random
Forest and creates the SQLite database, so a clean checkout boots straight into
a working platform.

The server binds to 127.0.0.1 only.  Every page and every API endpoint sits
behind a session login; nothing is exposed anonymously.  Before putting this on
a shared network, replace FLASK_SECRET_KEY, rotate the seeded demo accounts and
serve it through a real WSGI server behind TLS.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from flask import (
    Flask,
    abort,
    render_template,
    request,
    session,
)

import config
from routes import api, auth, history, main, predict
from services import database, ml_service
from utils import formatters

# POST endpoints that legitimately arrive without a form token.
CSRF_EXEMPT_ENDPOINTS: set[str] = set()


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=False)

    app.config.update(
        SECRET_KEY=config.SECRET_KEY,
        PERMANENT_SESSION_LIFETIME=timedelta(
            minutes=config.SESSION_LIFETIME_MINUTES
        ),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # Public deployments are served over TLS, so refuse to send the session
        # cookie over plain HTTP.
        SESSION_COOKIE_SECURE=config.IS_PRODUCTION,
        PREFERRED_URL_SCHEME="https" if config.IS_PRODUCTION else "http",
        JSON_SORT_KEYS=False,
        TEMPLATES_AUTO_RELOAD=config.DEBUG,
    )

    # Hosting platforms terminate TLS at a proxy and forward the original
    # scheme and host in X-Forwarded-* headers. Without this, redirects built by
    # url_for would come out as http:// and the client would be bounced.
    if config.IS_PRODUCTION:
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    # ------------------------------------------------------------------
    # Blueprints
    # ------------------------------------------------------------------
    app.register_blueprint(auth.bp)
    app.register_blueprint(main.bp)
    app.register_blueprint(predict.bp)
    app.register_blueprint(history.bp)
    app.register_blueprint(api.bp)

    formatters.register(app)

    # ------------------------------------------------------------------
    # Cross-site request forgery guard
    # ------------------------------------------------------------------
    @app.before_request
    def issue_csrf_token():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)

    @app.before_request
    def verify_csrf_token():
        if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
            return None
        if request.endpoint in CSRF_EXEMPT_ENDPOINTS:
            return None

        expected = session.get("csrf_token", "")
        supplied = (
            request.form.get("csrf_token")
            or request.headers.get("X-CSRF-Token", "")
        )
        if not expected or not secrets.compare_digest(str(supplied), expected):
            abort(400, description="Invalid or missing CSRF token.")
        return None

    # ------------------------------------------------------------------
    # Template context
    # ------------------------------------------------------------------
    @app.context_processor
    def inject_globals():
        alert_counts = {"unacknowledged": 0, "total": 0}
        if session.get("user_id"):
            try:
                alert_counts = database.alert_counts()
            except Exception:                       # noqa: BLE001
                pass

        static_dir = config.STATIC_DIR
        assets = {
            "primary_logo": (
                static_dir / config.BRAND["primary_logo_file"]
            ).exists(),
            "secondary_logo": (
                static_dir / config.BRAND["secondary_logo_file"]
            ).exists(),
            "background_video": (
                static_dir / config.BRAND["background_video_file"]
            ).exists(),
            "favicon_png": (static_dir / "img" / "favicon.png").exists(),
        }

        now = datetime.now()
        return {
            "brand": config.BRAND,
            "nav_sections": config.NAV_SECTIONS,
            "current_user": {
                "user_id": session.get("user_id"),
                "name": session.get("name"),
                "email": session.get("email"),
                "role": session.get("role"),
                "department": session.get("department"),
            },
            "alert_count": alert_counts.get("unacknowledged", 0),
            "alert_total": alert_counts.get("total", 0),
            "assets": assets,
            "csrf_token": session.get("csrf_token", ""),
            "server_date": formatters.clock_date(now),
            "server_time": formatters.clock_time(now),
            "server_long_date": formatters.long_date(now),
            "machine_count": len(config.MACHINES),
            "health_bands": config.HEALTH_BANDS,
            "priorities": config.PRIORITIES,
        }

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(error):
        return render_template(
            "error.html",
            code=400,
            title="Bad request",
            message=getattr(error, "description", "The request was rejected."),
        ), 400

    @app.errorhandler(403)
    def forbidden(error):
        return render_template(
            "error.html",
            code=403,
            title="Clearance required",
            message="Your account does not have access to that area.",
        ), 403

    @app.errorhandler(404)
    def not_found(error):
        return render_template(
            "error.html",
            code=404,
            title="Screen not found",
            message="That console screen does not exist.",
        ), 404

    @app.errorhandler(500)
    def server_error(error):
        return render_template(
            "error.html",
            code=500,
            title="Platform fault",
            message="An unexpected fault occurred. Check the server log.",
        ), 500

    return app


def bootstrap(verbose: bool = True) -> None:
    """Prepare datasets, model and database before the first request.

    Safe to call repeatedly: only missing artefacts are rebuilt, and the schema
    creation and user seeding are both idempotent. This is what makes the app
    survive a host with an ephemeral filesystem, where the database is gone
    after every restart.
    """
    problems = config.production_warnings()
    if problems:
        if config.IS_PRODUCTION:
            raise SystemExit(
                "Refusing to start in production with insecure configuration:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )
        if verbose:
            for problem in problems:
                print(f"[config] note: {problem}")

    ml_service.bootstrap(verbose=verbose)
    database.init_db()
    if verbose:
        card = ml_service.model_card()
        metrics = ml_service.metrics()
        print(f"[bootstrap] model    : {card['algorithm']} "
              f"({card['n_estimators']} trees), trained {card['trained_at']}")
        print(f"[bootstrap] test F1  : {metrics.get('f1')}%  "
              f"ROC-AUC {metrics.get('roc_auc')}%")
        print(f"[bootstrap] database : {config.DATABASE_PATH}")


app = create_app()


if __name__ == "__main__":
    bootstrap()

    # Bind to all interfaces only when a platform is hosting us; stay on
    # loopback for local development so nothing is exposed by accident.
    host = "0.0.0.0" if config.IS_PRODUCTION else "127.0.0.1"

    print()
    print("=" * 70)
    print(f"  {config.BRAND['project_title']}")
    print(f"  {config.BRAND['organisation']} - {config.BRAND['platform_name']}")
    print("=" * 70)
    print(f"  Environment : {config.APP_ENV}")
    print(f"  URL         : http://{host}:{config.PORT}")
    if config.SHOW_DEMO_KEYS:
        print(f"  Admin       : {config.ADMIN_EMAIL} / "
              f"{config.ADMIN_USER_ID} / {config.ADMIN_PASSWORD}")
        print(f"  Employee    : {config.EMPLOYEE_EMAIL} / "
              f"{config.EMPLOYEE_USER_ID} / {config.EMPLOYEE_PASSWORD}")
    else:
        print("  Credentials : set from environment (demo helper disabled)")
    print("  Stop        : Ctrl+C")
    print("=" * 70)
    print()
    app.run(host=host, port=config.PORT, debug=config.DEBUG)
