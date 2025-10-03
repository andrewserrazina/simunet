import os, json, uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# DB (Neon) if DATABASE_URL set; else JSON store
DB_AVAILABLE = False
try:
    import db
    DB_AVAILABLE = db.ensure_db()
except Exception:
    DB_AVAILABLE = False

SAVE_FILE = os.getenv("SIMUNET_SAVE_FILE", os.path.join(os.path.dirname(__file__), "data", "state.json"))

def load_state():
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"jobs": {}, "flights": {}, "telemetry": {}}

def save_state(state):
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

class TelemetryPoint(BaseModel):
    flight_id: str
    k: int
    lat: float
    lon: float
    alt: float = 0
    ts: Optional[str] = None  # ISO8601

app = FastAPI(title="SimuNet API", version="1.0.0")

# CORS
origins = [o.strip() for o in os.getenv("SIMUNET_CORS_ORIGINS", "*").split(",")] if os.getenv("SIMUNET_CORS_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def status():
    return {"ok": True, "db": DB_AVAILABLE}

@app.post("/dev/reseed")
def dev_reseed():
    if DB_AVAILABLE:
        with db.SessionLocal() as s:
            jid = str(uuid.uuid4())
            job = db.Job(job_id=jid)
            s.add(job)
            s.flush()
            s.add(db.Leg(seq=1, mode="truck", job_id=jid))
            s.commit()
            return {"ok": True, "job_id": jid}
    state = load_state()
    jid = str(uuid.uuid4())
    state["jobs"][jid] = {"job_id": jid, "legs": [{"seq": 1, "mode": "truck"}]}
    save_state(state)
    return {"ok": True, "job_id": jid}

@app.post("/flights")
def start_flight(assignment_id: str = Query(...), seq: int = Query(1), mode: str = Query("truck")):
    if DB_AVAILABLE:
        with db.SessionLocal() as s:
            job = s.get(db.Job, assignment_id)
            if job is None:
                job = db.Job(job_id=assignment_id)
                s.add(job)
                s.flush()
                s.add(db.Leg(seq=seq, mode=mode, job_id=assignment_id))
            fid = str(uuid.uuid4())
            s.add(db.Flight(flight_id=fid, assignment_id=assignment_id, seq=seq, mode=mode))
            s.commit()
            return {"ok": True, "flight_id": fid}
    state = load_state()
    if assignment_id not in state["jobs"]:
        state["jobs"][assignment_id] = {"job_id": assignment_id, "legs": [{"seq": seq, "mode": mode}]}
    fid = str(uuid.uuid4())
    state["flights"][fid] = {"flight_id": fid, "assignment_id": assignment_id, "seq": seq, "mode": mode}
    state["telemetry"].setdefault(fid, [])
    save_state(state)
    return {"ok": True, "flight_id": fid}

@app.post("/telemetry")
def post_telemetry(pt: TelemetryPoint):
    if DB_AVAILABLE:
        with db.SessionLocal() as s:
            row = s.query(db.Telemetry).filter_by(flight_id=pt.flight_id, k=pt.k).one_or_none()
            ts_val = None
            if pt.ts:
                try:
                    ts_val = datetime.fromisoformat(pt.ts.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    ts_val = None
            if row:
                row.lat = pt.lat
                row.lon = pt.lon
                row.alt = pt.alt
                row.ts  = ts_val
            else:
                s.add(db.Telemetry(flight_id=pt.flight_id, k=pt.k, lat=pt.lat, lon=pt.lon, alt=pt.alt, ts=ts_val))
            s.commit()
            return {"ok": True}
    state = load_state()
    lst = state["telemetry"].setdefault(pt.flight_id, [])
    d = pt.model_dump()
    idx = next((i for i, p in enumerate(lst) if p["k"] == pt.k), None)
    if idx is not None:
        lst[idx] = d
    else:
        lst.append(d)
    lst.sort(key=lambda r: r["k"])
    save_state(state)
    return {"ok": True}

@app.get("/telemetry/{flight_id}")
def get_telemetry(flight_id: str):
    if DB_AVAILABLE:
        with db.SessionLocal() as s:
            rows = (
                s.query(db.Telemetry)
                .filter_by(flight_id=flight_id)
                .order_by(db.Telemetry.k.asc())
                .all()
            )
            points = [{
                "flight_id": r.flight_id, "k": r.k, "lat": r.lat, "lon": r.lon, "alt": r.alt,
                "ts": r.ts.isoformat() if r.ts else None
            } for r in rows]
            return {"flight_id": flight_id, "points": points}
    state = load_state()
    return {"flight_id": flight_id, "points": state["telemetry"].get(flight_id, [])}
