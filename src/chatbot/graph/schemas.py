"""Validated contracts at model and graph boundaries."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chatbot.memory.schemas import MemoryCandidate
from chatbot.processes.schemas import ProcessControlAction, ProcessModel, ProcessView

RouteName = Literal["chat", "retrieve", "tools", "process"]
ToolName = Literal["calculator", "current_time", "knowledge_search"]


class RouteDecision(BaseModel):
    """Structured router output with consistent tool fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    route: RouteName
    tool_name: ToolName | None
    tool_input: str | None = Field(max_length=500)
    process_action: ProcessControlAction | None
    process_name: str | None = Field(
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )

    @model_validator(mode="after")
    def validate_tool_fields(self) -> "RouteDecision":
        if self.route == "tools":
            if self.tool_name is None:
                raise ValueError("Tool routes require a tool name")
            if self.tool_name != "current_time" and not (self.tool_input or "").strip():
                raise ValueError("This tool route requires tool input")
        elif self.tool_name is not None or self.tool_input is not None:
            raise ValueError("Only tool routes may include tool fields")
        if self.route == "process":
            if self.process_action is None:
                raise ValueError("Process routes require a process action")
            if self.process_action != "status" and self.process_name is None:
                raise ValueError("This process action requires a process name")
        elif self.process_action is not None or self.process_name is not None:
            raise ValueError("Only process routes may include process fields")
        return self

    def as_tool_arguments(self) -> dict[str, str]:
        if self.tool_name == "calculator":
            return {"expression": self.tool_input or ""}
        if self.tool_name == "current_time":
            return {"timezone": self.tool_input or "UTC"}
        if self.tool_name == "knowledge_search":
            return {"query": self.tool_input or ""}
        return {}


class ModelGateway(ProcessModel, Protocol):
    """Boundary around all provider calls so tests never need a real model."""

    def decide_route(
        self,
        message: str,
        process_context: str,
    ) -> RouteDecision | dict[str, Any]: ...

    def extract_memory(self, message: str) -> MemoryCandidate | dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ChatResult:
    response: str
    route: RouteName
    processes: tuple[ProcessView, ...]
