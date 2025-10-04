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
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

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

class Leg(Base):
    __tablename__ = "legs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="truck")
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.job_id"), index=True)
    job: Mapped["Job"] = relationship(back_populates="legs")


class JobOwner(Base):
    __tablename__ = "job_owners"
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.job_id"), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    job: Mapped["Job"] = relationship(back_populates="owner", uselist=False)


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
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("DB init: connected and tables ensured", flush=True)
        return True, None
    except Exception as e:
        import traceback
        print("DB init error:", e, flush=True)
        traceback.print_exc()
        return False, str(e)
