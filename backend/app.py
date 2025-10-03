import os, json, uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer  # if you’re using auth
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse


app=FastAPI()
@app.get('/status')
def s(): return {'ok':True,'db':False}

# Serve the frontend from the same container
FRONTEND_DIR = os.getenv(
    "SIMUNET_FRONTEND_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
)

assets_dir = os.path.join(FRONTEND_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/", include_in_schema=False)
def serve_index():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index):
        return {"ok": True, "db": DB_AVAILABLE, "hint": "Place frontend in /app/frontend or set SIMUNET_FRONTEND_DIR"}
    return FileResponse(index)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = os.path.join(FRONTEND_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404)

# SPA fallback: serve index.html for unknown non-API paths
@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    first = (full_path or "").split("/", 1)[0]
    if first in {"status","dev","flights","telemetry","auth","docs","openapi.json"}:
        # let FastAPI handle real API/docs routes
        raise HTTPException(status_code=404)
    candidate = os.path.join(FRONTEND_DIR, full_path)
    if os.path.isfile(candidate):
        return FileResponse(candidate)
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=404)


# Where the built/static frontend lives (we’ll copy it into the container at /app/frontend)
FRONTEND_DIR = os.getenv(
    "SIMUNET_FRONTEND_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
)

# Serve /assets/* straight from disk (icons, images, etc.)
assets_dir = os.path.join(FRONTEND_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/", include_in_schema=False)
def serve_index():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index):
        # still show something useful if the file isn’t there yet
        return {"ok": True, "db": DB_AVAILABLE, "hint": "Place frontend in /app/frontend or set SIMUNET_FRONTEND_DIR"}
    return FileResponse(index)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = os.path.join(FRONTEND_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404)
@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    # Don't intercept known API/doc routes
    first = (full_path or "").split("/", 1)[0]
    if first in {"status", "dev", "flights", "telemetry", "auth", "docs", "openapi.json"}:
        raise HTTPException(status_code=404)

    # Serve a real file if it exists (e.g., /assets/logo.svg), else serve index.html
    candidate = os.path.join(FRONTEND_DIR, full_path)
    if os.path.isfile(candidate):
        return FileResponse(candidate)
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=404)


@app.get("/", include_in_schema=False)
def home():
    db_state = "online" if DB_AVAILABLE else "offline"
    return HTMLResponse(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SimuNet API</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:2rem;color:#eaf2ff;background:#0b1020}
a{color:#7bb1ff;text-decoration:none} .box{padding:1rem;border:1px solid #273469;border-radius:12px;background:#0f1430;max-width:680px}
code{background:#0f1538;border:1px solid #273469;border-radius:6px;padding:2px 6px}</style></head>
<body>
  <div class="box">
    <h1>SimuNet API</h1>
    <p>Status: <strong>{db_state}</strong></p>
    <p>Try: <a href="/status">/status</a> · <a href="/docs">/docs</a></p>
  </div>
</body></html>""")

