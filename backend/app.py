"""FastAPI application exposing health and static file endpoints for SimuNet."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

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

USE_DB = bool(DB_AVAILABLE and getattr(db_module, "SessionLocal", None))


STATE_PATH = Path(os.path.join(os.path.dirname(__file__), "data", "state.json"))


def _isoformat(ts: datetime | str | None) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    value = ts
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_iso(ts: datetime | None = None) -> str:
    return _isoformat(ts or datetime.utcnow()) or datetime.utcnow().isoformat() + "Z"


def _load_state() -> dict[str, Any]:
    try:
        with STATE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {"jobs": {}, "flights": {}, "telemetry": {}}
    except json.JSONDecodeError:
        return {"jobs": {}, "flights": {}, "telemetry": {}}


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    tmp_path.replace(STATE_PATH)


@contextmanager
def _session():
    if not USE_DB:
        raise RuntimeError("Database not configured")
    session_factory = getattr(db_module, "SessionLocal", None)
    if session_factory is None:
        raise RuntimeError("Session factory unavailable")
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _serialize_point(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        return {
            "flight_id": point["flight_id"],
            "k": point["k"],
            "lat": point["lat"],
            "lon": point["lon"],
            "alt": point.get("alt", 0.0),
            "ts": point.get("ts"),
        }
    ts = _isoformat(getattr(point, "ts", None))
    return {
        "flight_id": point.flight_id,
        "k": point.k,
        "lat": point.lat,
        "lon": point.lon,
        "alt": point.alt,
        "ts": ts,
    }


class TelemetryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    flight_id: str = Field(..., min_length=1)
    k: int = Field(..., ge=0)
    lat: float
    lon: float
    alt: float = 0.0
    ts: datetime | None = None


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


@app.post("/dev/reseed", status_code=status.HTTP_201_CREATED)
def reseed() -> dict[str, Any]:
    """Create a demo job/assignment for quickstarts."""
    job_id = uuid.uuid4().hex[:12]
    legs = [
        {"seq": 1, "mode": "truck"},
        {"seq": 2, "mode": "air"},
    ]

    if USE_DB:
        with _session() as session:
            job = db_module.Job(job_id=job_id)
            session.add(job)
            for leg in legs:
                session.add(
                    db_module.Leg(
                        job_id=job_id,
                        seq=leg["seq"],
                        mode=leg["mode"],
                    )
                )
            session.flush()
    else:
        state = _load_state()
        state.setdefault("jobs", {})[job_id] = {
            "job_id": job_id,
            "legs": legs,
            "created_at": _now_iso(),
        }
        _save_state(state)

    return {"ok": True, "job_id": job_id, "legs": legs}


@app.post("/flights", status_code=status.HTTP_201_CREATED)
def create_flight(
    assignment_id: str = Query(..., min_length=1),
    seq: int = Query(1, ge=1),
    mode: str = Query("truck", min_length=1),
) -> dict[str, Any]:
    """Create a new flight for a given assignment."""

    flight_id = uuid.uuid4().hex
    if USE_DB:
        with _session() as session:
            session.add(
                db_module.Flight(
                    flight_id=flight_id,
                    assignment_id=assignment_id,
                    seq=seq,
                    mode=mode,
                )
            )
    else:
        state = _load_state()
        state.setdefault("flights", {})[flight_id] = {
            "flight_id": flight_id,
            "assignment_id": assignment_id,
            "seq": seq,
            "mode": mode,
            "created_at": _now_iso(),
        }
        _save_state(state)

    return {"ok": True, "flight_id": flight_id}


@app.post("/telemetry", status_code=status.HTTP_202_ACCEPTED)
def post_telemetry(payload: TelemetryPayload) -> dict[str, Any]:
    """Store telemetry points for a flight."""

    if USE_DB:
        with _session() as session:
            existing = (
                session.query(db_module.Telemetry)
                .filter_by(flight_id=payload.flight_id, k=payload.k)
                .one_or_none()
            )
            if existing:
                existing.lat = payload.lat
                existing.lon = payload.lon
                existing.alt = payload.alt
                existing.ts = payload.ts
            else:
                session.add(
                    db_module.Telemetry(
                        flight_id=payload.flight_id,
                        k=payload.k,
                        lat=payload.lat,
                        lon=payload.lon,
                        alt=payload.alt,
                        ts=payload.ts,
                    )
                )
    else:
        state = _load_state()
        bucket = state.setdefault("telemetry", {}).setdefault(payload.flight_id, [])
        new_point = {
            "flight_id": payload.flight_id,
            "k": payload.k,
            "lat": payload.lat,
            "lon": payload.lon,
            "alt": payload.alt,
            "ts": _isoformat(payload.ts),
        }
        replaced = False
        for idx, point in enumerate(bucket):
            if point.get("k") == payload.k:
                bucket[idx] = new_point
                replaced = True
                break
        if not replaced:
            bucket.append(new_point)
            bucket.sort(key=lambda item: item.get("k", 0))
        _save_state(state)

    return {"ok": True}


@app.get("/telemetry/{flight_id}")
def get_telemetry(flight_id: str) -> dict[str, Any]:
    """Return stored telemetry for a flight."""

    if USE_DB:
        with _session() as session:
            points = (
                session.query(db_module.Telemetry)
                .filter_by(flight_id=flight_id)
                .order_by(db_module.Telemetry.k.asc())
                .all()
            )
            serialized = [_serialize_point(p) for p in points]
    else:
        state = _load_state()
        stored = state.get("telemetry", {}).get(flight_id, [])
        serialized = [_serialize_point(p) for p in stored]

    return {"flight_id": flight_id, "points": serialized}


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
