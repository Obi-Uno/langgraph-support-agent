"""
Database layer for the Support Agent demo.

Uses SQLite by default (zero external dependency, no Supabase/hosted DB risk).
Swap DATABASE_URL in .env for Postgres/Neon if you want a hosted DB later.
"""
from datetime import datetime, timezone
import os

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./support_agent.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True)
    customer_email = Column(String, index=True, nullable=False)
    status = Column(String, nullable=False)  # placed, shipped, delivered, refunded
    amount = Column(Float, nullable=False)
    item = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_email = Column(String, index=True, nullable=False)
    order_id = Column(String, nullable=True)
    subject = Column(String, nullable=False)
    status = Column(String, default="open")  # open, in_progress, escalated, closed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    """
    Immutable-style audit trail of every agent decision.
    Mirrors the audit-logging requirement seen repeatedly in client postings.
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    event_type = Column(String, nullable=False)  # tool_call, guardrail_block, escalation, response
    detail = Column(Text, nullable=False)
    escalated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_event(db, session_id: str, event_type: str, detail: str, escalated: bool = False):
    entry = AuditLog(
        session_id=session_id,
        event_type=event_type,
        detail=detail,
        escalated=escalated,
    )
    db.add(entry)
    db.commit()
    return entry
