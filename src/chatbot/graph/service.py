"""Public synchronous interface for invoking the chatbot graph."""

import hashlib

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from chatbot.graph.nodes import SAFE_FAILURE_MESSAGE
from chatbot.graph.schemas import ChatResult
from chatbot.graph.state import GraphContext
from chatbot.processes.registry import ProcessRegistry
from chatbot.processes.schemas import ProcessAction, ProcessRecord, ProcessView


class InvalidProcessRequest(ValueError):
    """Raised when explicit process control references an unregistered process."""


class ChatbotService:
    def __init__(self, graph: CompiledStateGraph, processes: ProcessRegistry) -> None:
        self._graph = graph
        self._processes = processes

    def chat(
        self,
        *,
        subject: str,
        session_id: str,
        message: str,
        memory_consent: bool = False,
        correlation_id: str = "",
        process_action: ProcessAction = "auto",
        process_name: str | None = None,
    ) -> ChatResult:
        """Run one complete non-streaming turn."""

        if not subject or not session_id or not message.strip():
            raise ValueError("Subject, session ID, and message are required")
        if process_action in {"start", "continue", "switch", "cancel"}:
            if process_name is None:
                raise InvalidProcessRequest("This process action requires a process name")
        elif process_action == "auto" and process_name is not None:
            raise InvalidProcessRequest("A process name requires an explicit process action")
        if process_name is not None and process_name not in self._processes.names:
            raise InvalidProcessRequest("Unknown process name")
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
                "process_action": None,
                "process_name": None,
                "process_message": "",
                "process_should_dispatch": False,
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
                process_action=process_action,
                process_name=process_name,
            ),
        )
        route = state.get("route", "chat")
        if route not in {"chat", "retrieve", "tools", "process"}:
            route = "chat"
        active_process = state.get("active_process")
        process_views = []
        try:
            for raw_record in state.get("processes", {}).values():
                record = ProcessRecord.model_validate(raw_record)
                process_views.append(
                    ProcessView(
                        name=record.name,
                        status=record.status,
                        step=record.step,
                        active=record.name == active_process,
                    )
                )
        except Exception:
            process_views = []
        return ChatResult(
            response=state.get("response", SAFE_FAILURE_MESSAGE),
            route=route,
            processes=tuple(process_views),
        )


__all__ = ["SAFE_FAILURE_MESSAGE", "ChatbotService", "InvalidProcessRequest"]
