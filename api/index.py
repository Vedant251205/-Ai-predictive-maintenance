"""
Vercel serverless entrypoint.

Vercel's Python runtime imports this module and looks for a WSGI callable named
`app`. Two things have to be arranged before that import happens.

1. The project root must be on sys.path, because this file lives in api/ while
   the application package sits one level up.

2. The database must point at a writable location. A serverless filesystem is
   read-only apart from /tmp, so SQLite is placed there. /tmp is per-instance
   and ephemeral, which means stored predictions do not outlive a cold start.
   bootstrap() recreates the schema and re-seeds the accounts, so the site
   always comes up working; it simply does not accumulate history.

The dataset, the trained model and the precomputed kiln analytics are all
committed to the repository, so no generation or training happens here. Training
inside a request would blow the function timeout.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Writable location for SQLite on a serverless filesystem.
os.environ.setdefault("DATABASE_PATH", "/tmp/predictive_maintenance.db")

from app import app, bootstrap  # noqa: E402

bootstrap(verbose=False)

__all__ = ["app"]
