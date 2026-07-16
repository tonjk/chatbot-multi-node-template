"""Lifespan-managed SQLite or PostgreSQL LangGraph checkpointers."""

import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy.engine import make_url


class CheckpointerResource(AbstractContextManager[BaseCheckpointSaver[Any]]):
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._sqlite_connection: sqlite3.Connection | None = None
        self._postgres_context: AbstractContextManager[PostgresSaver] | None = None
        self._saver: BaseCheckpointSaver[Any] | None = None

    def __enter__(self) -> BaseCheckpointSaver[Any]:
        url = make_url(self._database_url)
        if url.drivername == "sqlite":
            database = url.database or ":memory:"
            if database != ":memory:":
                path = Path(database).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                database = str(path)
            connection = sqlite3.connect(database, check_same_thread=False)
            connection.execute("PRAGMA busy_timeout = 5000")
            if database != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
            saver = SqliteSaver(connection)
            saver.setup()
            self._sqlite_connection = connection
            self._saver = saver
            return saver
        if url.drivername in {"postgresql", "postgresql+psycopg"}:
            postgres_url = url.set(drivername="postgresql").render_as_string(hide_password=False)
            context = PostgresSaver.from_conn_string(postgres_url)
            saver = context.__enter__()
            saver.setup()
            self._postgres_context = context
            self._saver = saver
            return saver
        raise ValueError("DATABASE_URL must use SQLite or PostgreSQL")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._postgres_context is not None:
            self._postgres_context.__exit__(exc_type, exc_value, traceback)
        if self._sqlite_connection is not None:
            self._sqlite_connection.close()
        self._saver = None
        return None
