"""FastAPI application exposing health and static file endpoints for SimuNet."""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

app = FastAPI()

DB_AVAILABLE = False
DB_ERROR = None

try:
    from . import db as db_module  # type: ignore[import-self]
except ImportError:  # pragma: no cover - allows running without package context
    import db as db_module  # type: ignore[no-redef]

try:
    DB_AVAILABLE, DB_ERROR = db_module.ensure_db()
except Exception as exc:  # pragma: no cover - keep API responsive on DB failure
    DB_AVAILABLE, DB_ERROR = False, str(exc)


FRONTEND_DIR = os.getenv(
    "SIMUNET_FRONTEND_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")),
)

assets_dir = os.path.join(FRONTEND_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/status")
def status() -> dict[str, object]:
    """Simple health endpoint used by the frontend and smoke tests."""
    return {"ok": True, "db": DB_AVAILABLE, "error": DB_ERROR}


@app.get("/", include_in_schema=False)
def home():
    """Serve the SPA entrypoint if available, otherwise render a simple status page."""
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)

    db_state = "online" if DB_AVAILABLE else "offline"
    return HTMLResponse(
        """<!doctype html>
<html><head><meta charset="utf-8"><title>SimuNet API</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:2rem;color:#eaf2ff;background:#0b1020}
  a{color:#7bb1ff;text-decoration:none}
  .box{padding:1rem;border:1px solid #273469;border-radius:12px;background:#0f1430;max-width:680px}
  code{background:#0f1538;border:1px solid #273469;border-radius:6px;padding:2px 6px}
</style></head>
<body>
  <div class="box">
    <h1>SimuNet API</h1>
    <p>Status: <strong>{db_state}</strong></p>
    <p>Try: <a href="/status">/status</a> · <a href="/docs">/docs</a></p>
  </div>
</body></html>""".format(db_state=db_state)
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return the favicon if it exists on disk."""
    path = os.path.join(FRONTEND_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404)


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    """Serve static assets or fall back to the SPA entrypoint."""
    first = (full_path or "").split("/", 1)[0]
    if first in {"status", "dev", "flights", "telemetry", "auth", "docs", "openapi.json"}:
        raise HTTPException(status_code=404)

    candidate = os.path.join(FRONTEND_DIR, full_path)
    if os.path.isfile(candidate):
        return FileResponse(candidate)

    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)

    raise HTTPException(status_code=404)
