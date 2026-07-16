"""Structured JSON logging with request correlation context."""

import json
import logging
from contextvars import ContextVar, Token
from datetime import UTC, datetime

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_SAFE_EXTRA_FIELDS = {
    "duration_ms",
    "error_type",
    "event",
    "method",
    "node",
    "path",
    "route",
    "status_code",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "correlation_id": _correlation_id.get(),
        }
        for field in _SAFE_EXTRA_FIELDS:
            if field == "event":
                continue
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def configure_logging(level: str) -> None:
    """Configure chatbot loggers once without serializing request content."""

    application_logger = logging.getLogger("chatbot")
    application_logger.setLevel(level)
    application_logger.propagate = False
    if not any(getattr(handler, "_chatbot_json", False) for handler in application_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._chatbot_json = True  # type: ignore[attr-defined]
        application_logger.addHandler(handler)


def set_correlation_id(value: str) -> Token[str]:
    return _correlation_id.set(value)


def reset_correlation_id(token: Token[str]) -> None:
    _correlation_id.reset(token)


def get_correlation_id() -> str:
    return _correlation_id.get()
