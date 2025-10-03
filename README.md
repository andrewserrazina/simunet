# SimuNet — clean slate (Render Web Service + Neon)

**Drive. Fly. Deliver. Together.**  
This is a fresh, out-of-the-box SimuNet repo with:
- FastAPI backend (Neon Postgres if `DATABASE_URL` is set, JSON fallback if not)
- Minimal endpoints: `/status`, `/dev/reseed`, `/flights`, `/telemetry` (POST/GET)
- Zero-build frontend (`frontend/index.html`) that loads & animates telemetry
- Connectors (PowerShell & Python) to post telemetry
- Dockerfile at repo **root** (for Render **Web Service** / local Docker)
- `docker-compose.yml` for local dev (optional)

---

## Local (Python)

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Optional Neon:** set `DATABASE_URL` (SQLAlchemy/psycopg). Example:
```
export DATABASE_URL="postgresql+psycopg://USER:PASSWORD@YOUR-NEON-HOST/dbname?sslmode=require&channel_binding=require"
```

Open `frontend/index.html` in your browser. Use `?api=http://localhost:8000` if needed.

---

## Local (Docker)

```bash
docker compose up --build
# API on http://localhost:8000
```

---

## Deploy on Render (free) — Web Service (Docker runtime)

1) Push this repo to GitHub.
2) In Render: **New → Web Service → Build from GitHub**
   - **Root Directory**: `.` (repo root)
   - **Runtime**: Docker (uses root `Dockerfile`)
   - **Health Check Path**: `/status`
   - **Environment**:
     - `DATABASE_URL` = your Neon SQLAlchemy URL (with `+psycopg` & `sslmode=require`)
     - `SIMUNET_CORS_ORIGINS` = `https://YOUR-FRONTEND.onrender.com, https://YOUR-API.onrender.com`
3) Create a **Static Site** for the frontend:
   - **Root Directory**: `.`
   - **Build Command**: *(empty)*
   - **Publish Directory**: `frontend`
4) Open your site:
   - `https://YOUR-STATIC.onrender.com/?api=https://YOUR-API.onrender.com`

**Status check**
```
curl https://YOUR-API.onrender.com/status
# {"ok": true, "db": true}  ← true means Neon connected
```

---

## API Summary

- `GET /status` → `{ ok, db }`
- `POST /dev/reseed` → `{ ok, job_id }`
- `POST /flights?assignment_id=...&seq=1&mode=truck|air` → `{ ok, flight_id }`
- `POST /telemetry` (JSON) → `{ ok: true }`
- `GET /telemetry/{flight_id}` → `{ flight_id, points: [...] }`

---

## Connectors

**PowerShell** (Windows):
```powershell
# Get a flight_id first (see /dev/reseed then /flights)
connectors\common\Post-Telemetry.ps1 -FlightId <flight_id> -Base https://YOUR-API.onrender.com
```

**Python stub (MSFS; replace with SimConnect)**:
```bash
python connectors/msfs/simconnect_python_example.py
```

---

## Notes
- If `DATABASE_URL` is not set, the API uses a simple JSON store at `backend/data/state.json`.
- To avoid CORS issues, set `SIMUNET_CORS_ORIGINS` to your exact site URLs in Render.
- No external keys needed; frontend uses OpenStreetMap tiles.
