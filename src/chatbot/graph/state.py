"""Compact serializable graph state and non-persisted request context."""

from dataclasses import dataclass
from typing import Any

from langgraph.graph import MessagesState

from chatbot.graph.schemas import RouteName


class ChatState(MessagesState, total=False):
    route: RouteName
    tool_name: str | None
    tool_arguments: dict[str, Any]
    retrieved_context: str
    tool_result: str
    response: str
    error_code: str | None


@dataclass(frozen=True, slots=True)
class GraphContext:
    """Trusted request data supplied by the API and excluded from checkpoints."""

    subject: str
    session_id: str
    memory_consent: bool
    correlation_id: str
