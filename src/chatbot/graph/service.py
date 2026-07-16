"""Public synchronous interface for invoking the chatbot graph."""

import hashlib

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from chatbot.graph.nodes import SAFE_FAILURE_MESSAGE
from chatbot.graph.schemas import ChatResult
from chatbot.graph.state import GraphContext


class ChatbotService:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    def chat(
        self,
        *,
        subject: str,
        session_id: str,
        message: str,
        memory_consent: bool = False,
        correlation_id: str = "",
    ) -> ChatResult:
        """Run one complete non-streaming turn."""

        if not subject or not session_id or not message.strip():
            raise ValueError("Subject, session ID, and message are required")
        scoped_thread_id = hashlib.sha256(f"{subject}\0{session_id}".encode()).hexdigest()
        state = self._graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "route": "chat",
                "tool_name": None,
                "tool_arguments": {},
                "retrieved_context": "",
                "tool_result": "",
                "response": "",
                "error_code": None,
            },
            config={
                "configurable": {"thread_id": scoped_thread_id},
                "recursion_limit": 12,
            },
            context=GraphContext(
                subject=subject,
                session_id=session_id,
                memory_consent=bool(memory_consent),
                correlation_id=correlation_id,
            ),
        )
        route = state.get("route", "chat")
        if route not in {"chat", "retrieve", "tools"}:
            route = "chat"
        return ChatResult(response=state.get("response", SAFE_FAILURE_MESSAGE), route=route)


__all__ = ["SAFE_FAILURE_MESSAGE", "ChatbotService"]
