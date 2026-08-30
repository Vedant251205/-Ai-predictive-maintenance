"""
WSGI entrypoint for production hosting.

    gunicorn wsgi:app

Why this file exists rather than pointing gunicorn at app.py directly: the
dataset, trained model and SQLite database are prepared by bootstrap(), which
in app.py only runs under `__main__`. A WSGI server imports the module instead
of executing it, so without this shim the first request would arrive with no
database and no model.

Hosts such as Render give each instance an ephemeral filesystem, so the
database really is absent on every cold start. bootstrap() is idempotent and
re-creates the schema and seeds the demo accounts, which is what lets the site
come up working every time.
"""

from app import app, bootstrap

bootstrap(verbose=True)

# `app` is the WSGI callable gunicorn looks for.
__all__ = ["app"]
