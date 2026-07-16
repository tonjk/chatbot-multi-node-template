"""Owner-scoped durable-memory storage."""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import DateTime, Float, String, Text, create_engine, delete, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from chatbot.memory.schemas import MemoryCandidate, MemoryRecord

_MIN_CONFIDENCE = 0.75
_SENSITIVE_PATTERN = re.compile(
    r"\b(?:password|passcode|api[ -]?key|access[ -]?token|refresh[ -]?token|"
    r"private[ -]?key|secret|credit[ -]?card|card[ -]?number|cvv|pin|ssn|"
    r"social[ -]?security|passport|medical|diagnos\w*|medication|patient)\b",
    re.IGNORECASE,
)


class _Base(DeclarativeBase):
    pass


class _MemoryRow(_Base):
    __tablename__ = "durable_memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_key: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MemoryRepository:
    """Persist safe memories while enforcing consent and ownership."""

    def __init__(self, database_url: str) -> None:
        self._engine = self._create_engine(database_url)
        _Base.metadata.create_all(self._engine)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)

    def add(
        self,
        subject: str,
        session_id: str,
        candidate: MemoryCandidate,
        *,
        consent: bool,
    ) -> MemoryRecord | None:
        """Store one candidate only when every deterministic gate passes."""

        if not consent or not self._is_safe(candidate):
            return None
        if not subject or not session_id:
            return None

        row = _MemoryRow(
            id=str(uuid4()),
            owner_key=self._owner_key(subject),
            session_id=session_id,
            content=candidate.content.strip(),
            category=candidate.category,
            confidence=candidate.confidence,
            created_at=datetime.now(UTC),
        )
        with self._session() as session:
            session.add(row)
            session.commit()
        return self._to_record(row)

    def list_for_subject(self, subject: str, *, limit: int = 100) -> list[MemoryRecord]:
        """Return only the authenticated subject's memories, newest first."""

        bounded_limit = max(1, min(limit, 100))
        statement = (
            select(_MemoryRow)
            .where(_MemoryRow.owner_key == self._owner_key(subject))
            .order_by(_MemoryRow.created_at.desc())
            .limit(bounded_limit)
        )
        with self._session() as session:
            rows = session.scalars(statement).all()
        return [self._to_record(row) for row in rows]

    def delete(self, subject: str, memory_id: str) -> bool:
        """Delete by ID and owner in one predicate to prevent cross-owner access."""

        statement = delete(_MemoryRow).where(
            _MemoryRow.id == memory_id,
            _MemoryRow.owner_key == self._owner_key(subject),
        )
        with self._session() as session:
            result = session.execute(statement)
            session.commit()
        return bool(result.rowcount)

    def ping(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def close(self) -> None:
        self._engine.dispose()

    def _session(self) -> Session:
        return self._sessions()

    @staticmethod
    def _owner_key(subject: str) -> str:
        return hashlib.sha256(subject.encode()).hexdigest()

    @staticmethod
    def _is_safe(candidate: MemoryCandidate) -> bool:
        content = candidate.content.strip()
        return (
            candidate.should_store
            and not candidate.sensitive
            and candidate.confidence >= _MIN_CONFIDENCE
            and bool(content)
            and _SENSITIVE_PATTERN.search(content) is None
        )

    @staticmethod
    def _to_record(row: _MemoryRow) -> MemoryRecord:
        created_at = row.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return MemoryRecord(
            id=row.id,
            session_id=row.session_id,
            content=row.content,
            category=row.category,
            confidence=row.confidence,
            created_at=created_at,
        )

    @staticmethod
    def _create_engine(database_url: str) -> Engine:
        url = make_url(database_url)
        connect_args: dict[str, object] = {}
        if url.drivername == "sqlite":
            if url.database and url.database != ":memory:":
                Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
            connect_args["check_same_thread"] = False
        elif url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        return create_engine(url, connect_args=connect_args, pool_pre_ping=True)
