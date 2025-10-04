import os
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from sqlalchemy import (
    create_engine,
    text,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    MetaData,
    UniqueConstraint,
    Text,
    Boolean,
    inspect,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

def _normalize_url(url: str) -> str:
    """
    Ensure psycopg3 driver, SSL on, and remove channel_binding (can trip psycopg3).
    Works whether you pasted postgresql://... or postgresql+psycopg://...
    """
    if not url:
        return url
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q.setdefault("sslmode", "require")
    q.pop("channel_binding", None)
    url = urlunparse(p._replace(query=urlencode(q)))
    return url

DATABASE_URL = _normalize_url(os.getenv("DATABASE_URL", ""))

SIMUNET_CREATOR_EMAIL = "ops@simunet.local"

_engine = None
SessionLocal = None

class Base(DeclarativeBase):
    metadata = MetaData()

# ---------- MODELS ----------
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True, default=datetime.utcnow)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    legs: Mapped[list["Leg"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    owner: Mapped[Optional["JobOwner"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        single_parent=True,
        uselist=False,
    )
    detail: Mapped[Optional["JobDetail"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        single_parent=True,
        uselist=False,
    )

class Leg(Base):
    __tablename__ = "legs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="truck")
    origin_airport: Mapped[str | None] = mapped_column(String(16), nullable=True)
    destination_airport: Mapped[str | None] = mapped_column(String(16), nullable=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.job_id"), index=True)
    job: Mapped["Job"] = relationship(back_populates="legs")


class JobOwner(Base):
    __tablename__ = "job_owners"
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.job_id"), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    job: Mapped["Job"] = relationship(back_populates="owner", uselist=False)


class JobDetail(Base):
    __tablename__ = "job_details"

    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.job_id"), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    platform: Mapped[str] = mapped_column(String(128), nullable=False, default="Microsoft Flight Simulator")
    payload: Mapped[str] = mapped_column(String(255), nullable=False)
    weight_lbs: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    departure_airport: Mapped[str] = mapped_column(String(16), nullable=False)
    arrival_airport: Mapped[str] = mapped_column(String(16), nullable=False)
    deadline: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True, default=datetime.utcnow)
    assigned_to: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    team_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("teams.id"), nullable=True)
    airline_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("virtual_airlines.id"), nullable=True)

    job: Mapped["Job"] = relationship(back_populates="detail", uselist=False)
    team: Mapped[Optional["Team"]] = relationship(back_populates="jobs")
    airline: Mapped[Optional["VirtualAirline"]] = relationship(back_populates="jobs")


class Flight(Base):
    __tablename__ = "flights"
    flight_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    assignment_id: Mapped[str] = mapped_column(String(64), index=True)  # job_id
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="truck")

class Telemetry(Base):
    __tablename__ = "telemetry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flight_id: Mapped[str] = mapped_column(String(64), index=True)
    k: Mapped[int] = mapped_column(Integer, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    alt: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    __table_args__ = (UniqueConstraint("flight_id", "k", name="uq_flight_k"),)


class VirtualAirline(Base):
    __tablename__ = "virtual_airlines"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True, default=datetime.utcnow)

    teams: Mapped[list["Team"]] = relationship(back_populates="airline", cascade="all, delete-orphan")
    jobs: Mapped[list["JobDetail"]] = relationship(back_populates="airline")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True, default=datetime.utcnow)
    airline_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("virtual_airlines.id"), nullable=True)

    airline: Mapped[Optional[VirtualAirline]] = relationship(back_populates="teams")
    members: Mapped[list["TeamMembership"]] = relationship(back_populates="team", cascade="all, delete-orphan")
    jobs: Mapped[list[JobDetail]] = relationship(back_populates="team")


class TeamMembership(Base):
    __tablename__ = "team_memberships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(String(64), ForeignKey("teams.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    joined_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True, default=datetime.utcnow)

    team: Mapped[Team] = relationship(back_populates="members")

# ---------- INIT ----------
def _init_engine():
    global _engine, SessionLocal
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=_engine)

def ensure_db():
    """
    Create the engine, create tables, and run a sanity SELECT.
    Returns (True, None) on success, (False, 'error message') on failure.
    """
    try:
        _init_engine()
        Base.metadata.create_all(_engine)
        _apply_schema_patches()
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("DB init: connected and tables ensured", flush=True)
        return True, None
    except Exception as e:
        import traceback
        print("DB init error:", e, flush=True)
        traceback.print_exc()
        return False, str(e)


def _apply_schema_patches() -> None:
    if _engine is None:
        return

    try:
        inspector = inspect(_engine)
        tables = set(inspector.get_table_names())
    except Exception:
        return

    if "users" not in tables:
        return

    try:
        columns = {column["name"] for column in inspector.get_columns("users")}
    except Exception:
        return

    if "is_admin" in columns:
        return

    ddl = "ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT FALSE"
    update_admin = """
        UPDATE users
        SET is_admin = TRUE
        WHERE lower(email) = :email
    """

    with _engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(update_admin), {"email": SIMUNET_CREATOR_EMAIL})

    # Ensure new columns exist for collaborative missions without requiring a full migration tool.
    def _ensure_column(table: str, column: str, ddl_sql: str) -> None:
        try:
            existing = {col["name"] for col in inspector.get_columns(table)}
        except Exception:
            return
        if column in existing:
            return
        with _engine.begin() as conn:
            conn.execute(text(ddl_sql))

    if "job_details" in tables:
        _ensure_column(
            "job_details",
            "team_id",
            "ALTER TABLE job_details ADD COLUMN team_id VARCHAR(64) NULL",
        )
        _ensure_column(
            "job_details",
            "airline_id",
            "ALTER TABLE job_details ADD COLUMN airline_id VARCHAR(64) NULL",
        )

    if "legs" in tables:
        _ensure_column(
            "legs",
            "origin_airport",
            "ALTER TABLE legs ADD COLUMN origin_airport VARCHAR(16) NULL",
        )
        _ensure_column(
            "legs",
            "destination_airport",
            "ALTER TABLE legs ADD COLUMN destination_airport VARCHAR(16) NULL",
        )
