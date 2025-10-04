"""FastAPI application exposing health and static file endpoints for SimuNet."""

from __future__ import annotations

import json
import os
import sys
import uuid
import hashlib
import secrets
import types
import importlib
import importlib.util
import random
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, EmailStr

try:  # pragma: no cover - optional when running without SQLAlchemy
    from sqlalchemy.orm import joinedload
except Exception:  # pragma: no cover - keep optional dependency soft
    joinedload = None  # type: ignore[assignment]

app = FastAPI()

DB_AVAILABLE = False
DB_ERROR = None

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_db_module():
    """Import the database module regardless of deployment layout."""

    module_name = "backend.db"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass

    spec = importlib.util.spec_from_file_location(module_name, BACKEND_DIR / "db.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("Unable to load backend.db module")

    module = importlib.util.module_from_spec(spec)
    # Ensure both the package and module entries exist for downstream imports.
    package = sys.modules.get("backend")
    if package is None:
        package = types.ModuleType("backend")
        sys.modules["backend"] = package
    if not getattr(package, "__path__", None):
        package.__path__ = [str(BACKEND_DIR)]

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


db_module = _load_db_module()

try:
    DB_AVAILABLE, DB_ERROR = db_module.ensure_db()
except Exception as exc:  # pragma: no cover - keep API responsive on DB failure
    DB_AVAILABLE, DB_ERROR = False, str(exc)

USE_DB = bool(DB_AVAILABLE and getattr(db_module, "SessionLocal", None))


STATE_PATH = Path(os.path.join(os.path.dirname(__file__), "data", "state.json"))

SIMUNET_CREATOR_EMAIL = "ops@simunet.local"

MSFS_MISSIONS = [
    {
        "title": "Cascade Relief Hop",
        "payload": "Medical relief packages",
        "weight_lbs": (5200, 6800),
        "deadline_hours": (4, 9),
        "departure": "KSEA",
        "arrival": "KPDX",
        "notes": "Coordinate with Portland ground crew for hand-off on arrival.",
    },
    {
        "title": "Northern Lights Cargo",
        "payload": "Navigation beacons",
        "weight_lbs": (3800, 5400),
        "deadline_hours": (6, 12),
        "departure": "PAFA",
        "arrival": "PANC",
        "notes": "Watch for icing along the Alaska Range and maintain radio contact with Anchorage Center.",
    },
    {
        "title": "Island Supply Shuttle",
        "payload": "Island hospital supplies",
        "weight_lbs": (2600, 4200),
        "deadline_hours": (5, 10),
        "departure": "PHNL",
        "arrival": "PHOG",
        "notes": "Plan for strong trade winds on approach into Maui; deliver by dusk for coastal clinics.",
    },
    {
        "title": "Rocky Mountain Survey",
        "payload": "Aerial mapping equipment",
        "weight_lbs": (3100, 4700),
        "deadline_hours": (8, 14),
        "departure": "KDEN",
        "arrival": "KSLC",
        "notes": "Collect terrain imagery en route over the Uintas before descending into Salt Lake City.",
    },
    {
        "title": "Arctic Research Drop",
        "payload": "Scientific instruments",
        "weight_lbs": (4500, 6200),
        "deadline_hours": (7, 16),
        "departure": "BGTL",
        "arrival": "BGSF",
        "notes": "Limited daylight window — prioritize timely departure and monitor runway conditions in Greenland.",
    },
    {
        "title": "Coastal Weather Run",
        "payload": "Automated weather stations",
        "weight_lbs": (3300, 5100),
        "deadline_hours": (3, 7),
        "departure": "CYVR",
        "arrival": "CYYJ",
        "notes": "Distribute payload to Victoria field team; low ceilings expected over Strait of Georgia.",
    },
    {
        "title": "High Desert Calibration",
        "payload": "Survey calibration sensors",
        "weight_lbs": (3400, 5200),
        "deadline_hours": (5, 11),
        "departure": "KABQ",
        "arrival": "KPHX",
        "notes": "Calibrate the new desert survey array before the afternoon thermal activity picks up.",
    },
    {
        "title": "Great Lakes Ferry",
        "payload": "Freshwater research buoys",
        "weight_lbs": (2900, 4500),
        "deadline_hours": (4, 8),
        "departure": "KDTW",
        "arrival": "CYYZ",
        "notes": "Toronto team will meet you at the Cargo 6 stand — keep customs documents ready for quick transfer.",
    },
]

MSFS_DEFAULT_LEGS = [
    {"seq": 1, "mode": "flight"},
]


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
        return {"users": {}, "jobs": {}, "flights": {}, "telemetry": {}}
    except json.JSONDecodeError:
        return {"users": {}, "jobs": {}, "flights": {}, "telemetry": {}}


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    tmp_path.replace(STATE_PATH)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    iterations = 120_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${derived.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iteration_str, salt_hex, hash_hex = stored.split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iteration_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return secrets.compare_digest(candidate, expected)


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


def _serialize_job(job: Any) -> dict[str, Any]:
    def _status(assignee: str | None) -> str:
        return "claimed" if assignee else "open"

    if isinstance(job, dict):
        legs = job.get("legs") or []
        assigned_to = job.get("assigned_to")
        if assigned_to:
            assigned_to = _normalize_email(str(assigned_to))
        created_by = job.get("created_by")
        if created_by:
            created_by = _normalize_email(str(created_by))

        data = {
            "job_id": job.get("job_id"),
            "title": job.get("title"),
            "platform": job.get("platform"),
            "payload": job.get("payload"),
            "weight_lbs": job.get("weight_lbs"),
            "departure_airport": job.get("departure_airport"),
            "arrival_airport": job.get("arrival_airport"),
            "deadline": job.get("deadline"),
            "notes": job.get("notes"),
            "created_at": job.get("created_at"),
            "created_by": created_by,
            "assigned_to": assigned_to,
            "legs": legs,
        }
        data["status"] = _status(assigned_to)
        return data

    legs: list[dict[str, Any]] = []
    for leg in getattr(job, "legs", []) or []:
        legs.append({"seq": getattr(leg, "seq", 0), "mode": getattr(leg, "mode", "")})

    detail = getattr(job, "detail", None)
    created_by = getattr(detail, "created_by", None) if detail is not None else getattr(job, "created_by", None)
    if created_by:
        created_by = _normalize_email(str(created_by))

    assigned_to = None
    if detail is not None:
        assigned_to = getattr(detail, "assigned_to", None)
    owner = getattr(job, "owner", None)
    if owner is not None and getattr(owner, "email", None):
        assigned_to = getattr(owner, "email")
    if assigned_to:
        assigned_to = _normalize_email(str(assigned_to))

    created_at_source = getattr(detail, "created_at", None) if detail is not None else getattr(job, "created_at", None)

    return {
        "job_id": getattr(job, "job_id", None),
        "title": getattr(detail, "title", None) if detail is not None else getattr(job, "title", None),
        "platform": getattr(detail, "platform", None) if detail is not None else getattr(job, "platform", None),
        "payload": getattr(detail, "payload", None) if detail is not None else getattr(job, "payload", None),
        "weight_lbs": getattr(detail, "weight_lbs", None) if detail is not None else getattr(job, "weight_lbs", None),
        "departure_airport": getattr(detail, "departure_airport", None) if detail is not None else getattr(job, "departure_airport", None),
        "arrival_airport": getattr(detail, "arrival_airport", None) if detail is not None else getattr(job, "arrival_airport", None),
        "deadline": _isoformat(getattr(detail, "deadline", None)) if detail is not None else _isoformat(getattr(job, "deadline", None)),
        "notes": getattr(detail, "notes", None) if detail is not None else getattr(job, "notes", None),
        "created_at": _isoformat(created_at_source),
        "created_by": created_by,
        "assigned_to": assigned_to,
        "status": _status(assigned_to),
        "legs": legs,
    }


def _persist_job(
    *,
    job_id: str,
    creator_email: str,
    created_at: datetime,
    title: str,
    platform: str,
    payload_desc: str,
    weight_lbs: float | None,
    departure_airport: str,
    arrival_airport: str,
    deadline: datetime | None,
    notes: str | None,
    legs: list[dict[str, Any]],
    assigned_to: str | None = None,
) -> dict[str, Any]:
    """Persist a job in either the database or JSON fallback store."""

    normalized_creator = _normalize_email(creator_email)
    normalized_assignee = _normalize_email(assigned_to) if assigned_to else None
    departure = departure_airport.upper()
    arrival = arrival_airport.upper()
    deadline_dt = deadline
    if isinstance(deadline_dt, datetime) and deadline_dt.tzinfo is not None:
        deadline_dt = deadline_dt.astimezone(timezone.utc).replace(tzinfo=None)

    if USE_DB:
        with _session() as session:
            job = db_module.Job(job_id=job_id)
            session.add(job)

            detail_model = None
            if hasattr(db_module, "JobDetail"):
                detail_model = db_module.JobDetail(
                    job_id=job_id,
                    title=title,
                    platform=platform,
                    payload=payload_desc,
                    weight_lbs=weight_lbs,
                    departure_airport=departure,
                    arrival_airport=arrival,
                    deadline=deadline_dt,
                    notes=notes,
                    created_by=normalized_creator,
                    created_at=created_at,
                    assigned_to=normalized_assignee,
                )
                job.detail = detail_model
                session.add(detail_model)

            if normalized_assignee and hasattr(db_module, "JobOwner"):
                existing_owner = job.owner
                if existing_owner is None:
                    job.owner = db_module.JobOwner(job_id=job_id, email=normalized_assignee)
                else:
                    existing_owner.email = normalized_assignee

            for leg in legs or []:
                leg_model = db_module.Leg(
                    job=job,
                    seq=int(leg.get("seq", 1) or 1),
                    mode=str(leg.get("mode", "flight") or "flight"),
                )
                session.add(leg_model)

            session.flush()
            session.refresh(job)
            return _serialize_job(job)

    state = _load_state()
    jobs = state.setdefault("jobs", {})
    job_entry = {
        "job_id": job_id,
        "title": title,
        "platform": platform,
        "payload": payload_desc,
        "weight_lbs": weight_lbs,
        "departure_airport": departure,
        "arrival_airport": arrival,
        "deadline": _isoformat(deadline),
        "notes": notes,
        "created_at": _isoformat(created_at),
        "created_by": normalized_creator,
        "assigned_to": normalized_assignee,
        "legs": legs or [],
    }
    jobs[job_id] = job_entry
    _save_state(state)
    return _serialize_job(job_entry)


def _resolve_weight(weight_spec: Any) -> float:
    """Convert a mission weight specification into a concrete pound value."""

    default_low, default_high = 3000.0, 6500.0

    if isinstance(weight_spec, tuple) and len(weight_spec) == 2:
        low, high = weight_spec
        try:
            low_f = float(low)
            high_f = float(high)
        except (TypeError, ValueError):
            low_f, high_f = default_low, default_high
    elif isinstance(weight_spec, (int, float)):
        low_f = high_f = float(weight_spec)
    else:
        low_f, high_f = default_low, default_high

    if high_f < low_f:
        low_f, high_f = high_f, low_f

    if low_f == high_f:
        return round(low_f, 1)

    return round(random.uniform(low_f, high_f), 1)


def _resolve_deadline(now: datetime, hours_spec: Any) -> datetime:
    """Resolve a deadline relative to *now* using an hours specification."""

    default_range = (4, 12)
    if isinstance(hours_spec, tuple) and len(hours_spec) == 2:
        start, end = hours_spec
        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            start_i, end_i = default_range
    elif isinstance(hours_spec, (int, float)):
        start_i = end_i = int(hours_spec)
    else:
        start_i, end_i = default_range

    if end_i < start_i:
        start_i, end_i = end_i, start_i

    if start_i == end_i:
        offset = start_i
    else:
        offset = random.randint(start_i, end_i)

    offset = max(1, offset)
    return now + timedelta(hours=offset)


def _build_simunet_job(now: datetime | None = None) -> dict[str, Any]:
    """Create a randomized Microsoft Flight Simulator job blueprint."""

    current = now or datetime.utcnow()
    mission = random.choice(MSFS_MISSIONS)

    weight_value = _resolve_weight(mission.get("weight_lbs"))
    deadline = _resolve_deadline(current, mission.get("deadline_hours"))

    base_notes = mission.get("notes") or ""
    extra_note = "SimuNet Mission Control auto-generated assignment."
    notes = f"{base_notes}\n\n{extra_note}" if base_notes else extra_note

    legs = [dict(leg) for leg in MSFS_DEFAULT_LEGS]

    return {
        "title": mission.get("title", "Microsoft Flight Simulator Job"),
        "platform": "Microsoft Flight Simulator",
        "payload": mission.get("payload", "Logistics payload"),
        "weight_lbs": weight_value,
        "departure_airport": mission.get("departure", "KSEA"),
        "arrival_airport": mission.get("arrival", "CYVR"),
        "deadline": deadline,
        "notes": notes,
        "legs": legs,
    }


def _serialize_user(user: Any) -> dict[str, Any]:
    if isinstance(user, dict):
        email_value = user.get("email")
        normalized_email = _normalize_email(str(email_value)) if email_value else None
        is_admin_value = user.get("is_admin")
        if is_admin_value is None and normalized_email:
            is_admin_value = normalized_email == SIMUNET_CREATOR_EMAIL
        return {
            "id": user.get("id"),
            "email": email_value,
            "created_at": user.get("created_at"),
            "is_admin": bool(is_admin_value),
        }

    email_value = getattr(user, "email", None)
    normalized_email = _normalize_email(str(email_value)) if email_value else None
    is_admin_value = getattr(user, "is_admin", None)
    if is_admin_value is None and normalized_email:
        is_admin_value = normalized_email == SIMUNET_CREATOR_EMAIL

    return {
        "id": getattr(user, "id", None),
        "email": email_value,
        "created_at": _isoformat(getattr(user, "created_at", None)),
        "is_admin": bool(is_admin_value),
    }


def _user_is_admin(user: Any, email: str | None = None) -> bool:
    if isinstance(user, dict):
        flag = user.get("is_admin")
        if flag is not None:
            return bool(flag)
        candidate = user.get("email")
        if candidate:
            return _normalize_email(str(candidate)) == SIMUNET_CREATOR_EMAIL
        if email:
            return _normalize_email(email) == SIMUNET_CREATOR_EMAIL
        return False

    attr = getattr(user, "is_admin", None)
    if attr is not None:
        return bool(attr)

    candidate = email or getattr(user, "email", None)
    if candidate:
        return _normalize_email(str(candidate)) == SIMUNET_CREATOR_EMAIL
    return False


def _require_admin(actor_email: str | None) -> str:
    if not actor_email:
        raise HTTPException(status_code=401, detail="Admin privileges required")

    normalized = _normalize_email(str(actor_email))

    if USE_DB:
        with _session() as session:
            user = session.query(db_module.User).filter_by(email=normalized).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")
            if not _user_is_admin(user, normalized):
                raise HTTPException(status_code=403, detail="Admin privileges required")
    else:
        state = _load_state()
        users = state.setdefault("users", {})
        user = users.get(normalized)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if not _user_is_admin(user, normalized):
            raise HTTPException(status_code=403, detail="Admin privileges required")

    return normalized


def _count_admins_db(session: Any) -> int:
    if not USE_DB:
        return 0
    user_model = getattr(db_module, "User", None)
    if user_model is None:
        return 0
    admin_attr = getattr(user_model, "is_admin", None)
    if admin_attr is None:
        return 0
    return session.query(user_model).filter(admin_attr.is_(True)).count()


def _count_admins_state(users: dict[str, Any]) -> int:
    return sum(1 for email, data in users.items() if _user_is_admin(data, email))


class TelemetryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    flight_id: str = Field(..., min_length=1)
    k: int = Field(..., ge=0)
    lat: float
    lon: float
    alt: float = 0.0
    ts: datetime | None = None


class AuthPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(..., min_length=8)


class JobLegPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int = Field(1, ge=1)
    mode: str = Field("flight", min_length=1, max_length=32)


class JobCreatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=3, max_length=255)
    platform: str = Field("Microsoft Flight Simulator", min_length=3, max_length=128)
    payload: str = Field(..., min_length=3, max_length=255)
    weight_lbs: float | None = Field(default=None, ge=0)
    departure_airport: str = Field(..., min_length=3, max_length=16)
    arrival_airport: str = Field(..., min_length=3, max_length=16)
    deadline: datetime
    notes: str | None = Field(default=None, max_length=600)
    created_by: EmailStr
    assigned_to: EmailStr | None = None
    legs: list[JobLegPayload] = Field(default_factory=list)


class JobAutoPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=1, ge=1, le=10)


class JobClaimPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr


class ReseedPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: EmailStr | None = None


class AdminUserUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_admin: bool | None = None
    password: str | None = Field(default=None, min_length=8)


FRONTEND_DIR = os.getenv(
    "SIMUNET_FRONTEND_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")),
)

assets_dir = os.path.join(FRONTEND_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/status")
def get_status() -> dict[str, object]:
    """Simple health endpoint used by the frontend and smoke tests."""
    return {"ok": True, "db": DB_AVAILABLE, "error": DB_ERROR}


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(payload: AuthPayload) -> dict[str, Any]:
    email = _normalize_email(payload.email)
    password = payload.password

    if USE_DB:
        with _session() as session:
            existing = session.query(db_module.User).filter_by(email=email).one_or_none()
            if existing:
                raise HTTPException(status_code=409, detail="Email already registered")
            is_admin = email == SIMUNET_CREATOR_EMAIL
            if not is_admin and hasattr(db_module.User, "is_admin"):
                is_admin = _count_admins_db(session) == 0

            user_kwargs = {
                "email": email,
                "hashed_password": _hash_password(password),
                "created_at": datetime.utcnow(),
            }
            if hasattr(db_module.User, "is_admin"):
                user_kwargs["is_admin"] = bool(is_admin)

            user = db_module.User(**user_kwargs)
            session.add(user)
            session.flush()
            return {"ok": True, "user": _serialize_user(user)}

    state = _load_state()
    users = state.setdefault("users", {})
    if email in users:
        raise HTTPException(status_code=409, detail="Email already registered")
    created_at = _now_iso()
    user_id = uuid.uuid4().hex[:12]
    is_admin = email == SIMUNET_CREATOR_EMAIL or _count_admins_state(users) == 0
    users[email] = {
        "id": user_id,
        "email": email,
        "hashed_password": _hash_password(password),
        "created_at": created_at,
        "is_admin": bool(is_admin),
    }
    _save_state(state)
    return {"ok": True, "user": _serialize_user(users[email])}


@app.post("/auth/login")
def login(payload: AuthPayload) -> dict[str, Any]:
    email = _normalize_email(payload.email)
    password = payload.password

    if USE_DB:
        with _session() as session:
            user = session.query(db_module.User).filter_by(email=email).one_or_none()
            if user is None or not _verify_password(password, user.hashed_password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            if getattr(user, "created_at", None) is None:
                user.created_at = datetime.utcnow()
                session.flush()
            if email == SIMUNET_CREATOR_EMAIL and hasattr(user, "is_admin") and not _user_is_admin(user, email):
                user.is_admin = True  # type: ignore[assignment]
                session.flush()
            return {"ok": True, "user": _serialize_user(user)}

    state = _load_state()
    users = state.setdefault("users", {})
    stored = users.get(email)
    if not stored or not _verify_password(password, stored.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if "is_admin" not in stored:
        stored["is_admin"] = _user_is_admin(stored, email)
        users[email] = stored
        _save_state(state)
    return {"ok": True, "user": _serialize_user(stored)}


@app.post("/dev/reseed", status_code=status.HTTP_201_CREATED)
def reseed(payload: ReseedPayload | None = None) -> dict[str, Any]:
    """Create a demo job/assignment for quickstarts."""
    job_id = uuid.uuid4().hex[:12]
    created_at = datetime.utcnow()
    legs = [
        {"seq": 1, "mode": "flight"},
    ]

    owner_email = None
    if payload and payload.email:
        owner_email = _normalize_email(str(payload.email))

    if owner_email:
        if USE_DB:
            with _session() as session:
                user = (
                    session.query(db_module.User)
                    .filter_by(email=owner_email)
                    .one_or_none()
                )
                if user is None:
                    raise HTTPException(status_code=404, detail="User not found")
        else:
            state = _load_state()
            users = state.setdefault("users", {})
            if owner_email not in users:
                raise HTTPException(status_code=404, detail="User not found")

    job_data = _persist_job(
        job_id=job_id,
        creator_email=owner_email or SIMUNET_CREATOR_EMAIL,
        created_at=created_at,
        title="MSFS Relief Flight",
        platform="Microsoft Flight Simulator",
        payload_desc="Emergency medical supplies",
        weight_lbs=7200.0,
        departure_airport="KSEA",
        arrival_airport="CYVR",
        deadline=created_at + timedelta(hours=6),
        notes="Deliver the payload before the deadline to keep regional hospitals stocked.",
        legs=legs,
        assigned_to=owner_email,
    )

    return {
        "ok": True,
        "job_id": job_data.get("job_id"),
        "job": job_data,
    }


@app.post("/jobs", status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreatePayload) -> dict[str, Any]:
    """Create a new Microsoft Flight Simulator job."""

    creator_email = _normalize_email(str(payload.created_by))
    assignee_email = _normalize_email(str(payload.assigned_to)) if payload.assigned_to else None

    if USE_DB:
        with _session() as session:
            user = session.query(db_module.User).filter_by(email=creator_email).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")
            if assignee_email:
                assignee = session.query(db_module.User).filter_by(email=assignee_email).one_or_none()
                if assignee is None:
                    raise HTTPException(status_code=404, detail="Assignee not found")
    else:
        state = _load_state()
        users = state.setdefault("users", {})
        if creator_email not in users:
            raise HTTPException(status_code=404, detail="User not found")
        if assignee_email and assignee_email not in users:
            raise HTTPException(status_code=404, detail="Assignee not found")

    legs = [
        {"seq": leg.seq, "mode": leg.mode}
        for leg in (payload.legs or [])
    ]
    if not legs:
        legs = [{"seq": 1, "mode": "flight"}]

    job_id = uuid.uuid4().hex[:12]
    created_at = datetime.utcnow()

    job_data = _persist_job(
        job_id=job_id,
        creator_email=creator_email,
        created_at=created_at,
        title=payload.title.strip(),
        platform=payload.platform.strip(),
        payload_desc=payload.payload.strip(),
        weight_lbs=payload.weight_lbs,
        departure_airport=payload.departure_airport.strip(),
        arrival_airport=payload.arrival_airport.strip(),
        deadline=payload.deadline,
        notes=payload.notes.strip() if payload.notes else None,
        legs=legs,
        assigned_to=assignee_email,
    )

    return {"ok": True, "job": job_data}


@app.post("/jobs/generate", status_code=status.HTTP_201_CREATED)
def auto_generate_jobs(payload: JobAutoPayload | None = None) -> dict[str, Any]:
    """Create SimuNet-authored jobs without manual input."""

    count = 1
    if payload is not None:
        count = int(payload.count)

    created_jobs: list[dict[str, Any]] = []
    for _ in range(count):
        job_id = uuid.uuid4().hex[:12]
        created_at = datetime.utcnow()
        blueprint = _build_simunet_job(created_at)
        job = _persist_job(
            job_id=job_id,
            creator_email=SIMUNET_CREATOR_EMAIL,
            created_at=created_at,
            title=blueprint["title"],
            platform=blueprint["platform"],
            payload_desc=blueprint["payload"],
            weight_lbs=blueprint["weight_lbs"],
            departure_airport=blueprint["departure_airport"],
            arrival_airport=blueprint["arrival_airport"],
            deadline=blueprint["deadline"],
            notes=blueprint["notes"],
            legs=blueprint["legs"],
            assigned_to=None,
        )
        created_jobs.append(job)

    return {"ok": True, "jobs": created_jobs, "count": len(created_jobs)}


@app.post("/jobs/{job_id}/claim")
def claim_job(job_id: str, payload: JobClaimPayload) -> dict[str, Any]:
    """Assign a job to a specific user."""

    claimer = _normalize_email(str(payload.email))

    if USE_DB:
        with _session() as session:
            user = session.query(db_module.User).filter_by(email=claimer).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")

            job = session.query(db_module.Job).filter_by(job_id=job_id).one_or_none()
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")

            existing_assignee = None
            owner_cls = getattr(db_module, "JobOwner", None)
            if owner_cls is not None and job.owner is not None:
                existing_assignee = _normalize_email(str(job.owner.email))

            detail_model = getattr(job, "detail", None)
            if existing_assignee is None and detail_model is not None and getattr(detail_model, "assigned_to", None):
                existing_assignee = _normalize_email(str(detail_model.assigned_to))

            if existing_assignee and existing_assignee != claimer:
                raise HTTPException(status_code=409, detail="Job already claimed")

            if owner_cls is not None:
                if job.owner is None:
                    job.owner = owner_cls(job_id=job_id, email=claimer)
                else:
                    job.owner.email = claimer

            if detail_model is not None:
                detail_model.assigned_to = claimer
            elif hasattr(db_module, "JobDetail"):
                detail_cls = getattr(db_module, "JobDetail")
                existing_detail = session.query(detail_cls).filter_by(job_id=job_id).one_or_none()
                if existing_detail is not None:
                    existing_detail.assigned_to = claimer

            session.flush()
            session.refresh(job)
            return {"ok": True, "job": _serialize_job(job)}

    state = _load_state()
    users = state.setdefault("users", {})
    if claimer not in users:
        raise HTTPException(status_code=404, detail="User not found")

    jobs = state.setdefault("jobs", {})
    job_entry = jobs.get(job_id)
    if not job_entry:
        raise HTTPException(status_code=404, detail="Job not found")

    existing_assignee = job_entry.get("assigned_to")
    if existing_assignee:
        existing_assignee_norm = _normalize_email(str(existing_assignee))
        if existing_assignee_norm and existing_assignee_norm != claimer:
            raise HTTPException(status_code=409, detail="Job already claimed")

    job_entry["assigned_to"] = claimer
    jobs[job_id] = job_entry
    _save_state(state)
    return {"ok": True, "job": _serialize_job(job_entry)}


@app.get("/jobs")
def list_jobs(email: EmailStr | None = Query(default=None)) -> dict[str, Any]:
    """Return available jobs and the caller's claimed jobs."""

    filter_email = _normalize_email(str(email)) if email else None
    available: list[dict[str, Any]] = []
    mine: list[dict[str, Any]] = []

    def _sort_key(item: dict[str, Any]) -> tuple[str, str]:
        deadline = item.get("deadline") or ""
        created = item.get("created_at") or ""
        return (deadline or "", created or "")

    if USE_DB:
        with _session() as session:
            query = session.query(db_module.Job)
            if joinedload is not None:
                query = query.options(joinedload(db_module.Job.legs))
                if hasattr(db_module, "JobOwner"):
                    query = query.options(joinedload(db_module.Job.owner))
                if hasattr(db_module, "JobDetail"):
                    query = query.options(joinedload(db_module.Job.detail))

            jobs = query.order_by(db_module.Job.job_id.desc()).all()
            for job in jobs:
                serialized = _serialize_job(job)
                assignee = serialized.get("assigned_to")
                if assignee:
                    if filter_email and assignee == filter_email:
                        mine.append(serialized)
                    continue
                available.append(serialized)
    else:
        state = _load_state()
        stored = state.get("jobs", {})
        for job in stored.values():
            serialized = _serialize_job(job)
            assignee = serialized.get("assigned_to")
            if assignee:
                if filter_email and assignee == filter_email:
                    mine.append(serialized)
                continue
            available.append(serialized)

    available.sort(key=_sort_key)
    mine.sort(key=_sort_key)

    response = {"available": available, "mine": mine if filter_email else []}
    return response


@app.get("/admin/jobs")
def admin_list_jobs(actor: EmailStr = Query(..., alias="actor")) -> dict[str, Any]:
    """Return every job for administrator dashboards."""

    _require_admin(str(actor))

    jobs: list[dict[str, Any]] = []
    if USE_DB:
        with _session() as session:
            query = session.query(db_module.Job)
            if joinedload is not None:
                query = query.options(joinedload(db_module.Job.legs))
                if hasattr(db_module, "JobOwner"):
                    query = query.options(joinedload(db_module.Job.owner))
                if hasattr(db_module, "JobDetail"):
                    query = query.options(joinedload(db_module.Job.detail))
            records = query.order_by(db_module.Job.job_id.desc()).all()
            jobs = [_serialize_job(job) for job in records]
    else:
        state = _load_state()
        stored = state.get("jobs", {})
        jobs = [_serialize_job(job) for job in stored.values()]
    jobs.sort(key=lambda item: (item.get("created_at") or "", item.get("job_id") or ""), reverse=True)
    return {"ok": True, "jobs": jobs}


@app.delete("/admin/jobs/{job_id}")
def admin_delete_job(job_id: str, actor: EmailStr = Query(..., alias="actor")) -> dict[str, Any]:
    """Delete a job regardless of claim status."""

    _require_admin(str(actor))

    if USE_DB:
        with _session() as session:
            job = session.query(db_module.Job).filter_by(job_id=job_id).one_or_none()
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            session.delete(job)
            session.flush()
            return {"ok": True, "job_id": job_id}

    state = _load_state()
    jobs = state.setdefault("jobs", {})
    job_entry = jobs.pop(job_id, None)
    if job_entry is None:
        raise HTTPException(status_code=404, detail="Job not found")
    _save_state(state)
    return {"ok": True, "job_id": job_id, "job": _serialize_job(job_entry)}


@app.get("/admin/users")
def admin_list_users(actor: EmailStr = Query(..., alias="actor")) -> dict[str, Any]:
    """Return all user profiles."""

    _require_admin(str(actor))

    if USE_DB:
        with _session() as session:
            records = (
                session.query(db_module.User)
                .order_by(db_module.User.email.asc())
                .all()
            )
            users = [_serialize_user(record) for record in records]
            return {"ok": True, "users": users}

    state = _load_state()
    stored_users = state.get("users", {})
    users = []
    for email, data in sorted(stored_users.items()):
        if "email" not in data:
            data = dict(data)
            data["email"] = email
        users.append(_serialize_user(data))
    return {"ok": True, "users": users}


@app.patch("/admin/users/{email}")
def admin_update_user(
    email: str,
    payload: AdminUserUpdatePayload,
    actor: EmailStr = Query(..., alias="actor"),
) -> dict[str, Any]:
    """Update admin status or password for a user profile."""

    actor_email = _require_admin(str(actor))
    target_email = _normalize_email(email)

    if USE_DB:
        with _session() as session:
            user = session.query(db_module.User).filter_by(email=target_email).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")

            updated = False
            if payload.is_admin is not None and hasattr(user, "is_admin"):
                desired = bool(payload.is_admin)
                current_admin = _user_is_admin(user, target_email)
                if desired != current_admin:
                    if not desired and _count_admins_db(session) <= 1 and current_admin:
                        raise HTTPException(status_code=400, detail="Cannot remove last admin")
                    user.is_admin = desired  # type: ignore[assignment]
                    updated = True

            if payload.password:
                user.hashed_password = _hash_password(payload.password)
                updated = True

            if updated:
                session.flush()

            return {"ok": True, "user": _serialize_user(user)}

    state = _load_state()
    users = state.setdefault("users", {})
    user = users.get(target_email)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    updated = False
    if payload.is_admin is not None:
        desired = bool(payload.is_admin)
        current_admin = _user_is_admin(user, target_email)
        if desired != current_admin:
            if not desired and _count_admins_state(users) <= 1 and current_admin:
                raise HTTPException(status_code=400, detail="Cannot remove last admin")
            user["is_admin"] = desired
            updated = True

    if payload.password:
        user["hashed_password"] = _hash_password(payload.password)
        updated = True

    if updated:
        users[target_email] = user
        _save_state(state)

    return {"ok": True, "user": _serialize_user(user)}


@app.delete("/admin/users/{email}")
def admin_delete_user(email: str, actor: EmailStr = Query(..., alias="actor")) -> dict[str, Any]:
    """Remove a user account and release related assignments."""

    actor_email = _require_admin(str(actor))
    target_email = _normalize_email(email)

    if USE_DB:
        with _session() as session:
            user = session.query(db_module.User).filter_by(email=target_email).one_or_none()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")

            if _user_is_admin(user, target_email) and _count_admins_db(session) <= 1:
                raise HTTPException(status_code=400, detail="Cannot remove last admin")

            owner_cls = getattr(db_module, "JobOwner", None)
            if owner_cls is not None:
                owners = session.query(owner_cls).filter_by(email=target_email).all()
                for owner in owners:
                    session.delete(owner)

            detail_cls = getattr(db_module, "JobDetail", None)
            if detail_cls is not None:
                details = session.query(detail_cls).filter_by(assigned_to=target_email).all()
                for detail in details:
                    detail.assigned_to = None

            session.delete(user)
            session.flush()

            if target_email == actor_email:
                return {"ok": True, "deleted": target_email, "self": True}
            return {"ok": True, "deleted": target_email}

    state = _load_state()
    users = state.setdefault("users", {})
    user = users.get(target_email)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if _user_is_admin(user, target_email) and _count_admins_state(users) <= 1:
        raise HTTPException(status_code=400, detail="Cannot remove last admin")

    users.pop(target_email, None)

    jobs = state.setdefault("jobs", {})
    for job in jobs.values():
        assignee = job.get("assigned_to")
        if assignee and _normalize_email(str(assignee)) == target_email:
            job["assigned_to"] = None

    _save_state(state)
    return {"ok": True, "deleted": target_email}


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
