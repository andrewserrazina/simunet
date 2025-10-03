import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, String, Integer, Float, DateTime, ForeignKey, MetaData, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

class Base(DeclarativeBase):
    metadata = MetaData()

class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    legs: Mapped[list["Leg"]] = relationship(back_populates="job", cascade="all, delete-orphan")

class Leg(Base):
    __tablename__ = "legs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="truck")
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.job_id"), index=True)
    job: Mapped["Job"] = relationship(back_populates="legs")

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

def ensure_db():
    if _engine is None:
        return False
    Base.metadata.create_all(_engine)
    return True
