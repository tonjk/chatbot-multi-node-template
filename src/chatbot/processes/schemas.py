"""Validated contracts shared by the orchestrator and process plug-ins."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

ProcessAction = Literal["auto", "start", "continue", "switch", "cancel", "status"]
ProcessControlAction = Literal["start", "continue", "switch", "cancel", "status"]
ProcessStatus = Literal["waiting", "suspended", "completed", "cancelled", "failed"]

_PROCESS_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
_MAX_PAYLOAD_BYTES = 12_000
_BoundedInteger = Annotated[int, Field(ge=-1_000_000_000, le=1_000_000_000)]
_ColorName = Annotated[str, Field(min_length=1, max_length=40)]


class ProcessDirective(BaseModel):
    """One bounded process target in an AI-selected multi-process turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    process_action: Literal["start", "continue"]
    process_name: str = Field(max_length=64, pattern=_PROCESS_NAME_PATTERN)


class NumberAction(BaseModel):
    """One ordered mutation extracted from a number-counter message."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["add", "remove", "edit"]
    value: _BoundedInteger
    replacement: _BoundedInteger | None

    @model_validator(mode="after")
    def validate_replacement(self) -> "NumberAction":
        if self.action == "edit" and self.replacement is None:
            raise ValueError("Edit actions require a replacement")
        if self.action != "edit" and self.replacement is not None:
            raise ValueError("Only edit actions may include a replacement")
        return self


class NumberExtraction(BaseModel):
    """Bounded ordered number mutations extracted from one user message."""

    model_config = ConfigDict(extra="forbid", strict=True)

    actions: list[NumberAction] = Field(max_length=20)


class ColorAction(BaseModel):
    """One ordered mutation extracted from a color-note message."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["add", "remove", "edit"]
    value: _ColorName
    replacement: _ColorName | None

    @model_validator(mode="after")
    def validate_replacement(self) -> "ColorAction":
        if self.action == "edit" and self.replacement is None:
            raise ValueError("Edit actions require a replacement")
        if self.action != "edit" and self.replacement is not None:
            raise ValueError("Only edit actions may include a replacement")
        return self


class ColorExtraction(BaseModel):
    """Bounded ordered color mutations extracted from one user message."""

    model_config = ConfigDict(extra="forbid", strict=True)

    actions: list[ColorAction] = Field(max_length=20)


class ProcessModel(Protocol):
    """Minimal model boundary available to model-backed process steps."""

    def generate(self, messages: list[BaseMessage]) -> str: ...

    def extract_numbers(self, message: str) -> NumberExtraction | dict[str, Any]: ...

    def extract_colors(self, message: str) -> ColorExtraction | dict[str, Any]: ...


class ProcessRecord(BaseModel):
    """Bounded checkpoint-safe state owned by one registered process."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(pattern=_PROCESS_NAME_PATTERN, max_length=64)
    status: ProcessStatus
    step: str = Field(min_length=1, max_length=64, pattern=_PROCESS_NAME_PATTERN)
    payload: dict[str, Any]
    prompt: str = Field(max_length=2_000)
    result: str = Field(default="", max_length=4_000)

    @model_validator(mode="after")
    def validate_checkpoint_payload(self) -> "ProcessRecord":
        try:
            encoded = json.dumps(
                self.payload,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        except (TypeError, ValueError) as error:
            raise ValueError("Process payload must be JSON serializable") from error
        if len(encoded) > _MAX_PAYLOAD_BYTES:
            raise ValueError("Process payload is too large")
        return self


class ProcessView(BaseModel):
    """Safe process metadata returned to API callers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(pattern=_PROCESS_NAME_PATTERN, max_length=64)
    status: ProcessStatus
    step: str = Field(min_length=1, max_length=64, pattern=_PROCESS_NAME_PATTERN)
    active: bool


class ProcessGraphState(TypedDict, total=False):
    """Common adapter state used when invoking a stateless child graph."""

    name: str
    status: ProcessStatus
    step: str
    payload: dict[str, Any]
    prompt: str
    result: str
    message: str


@dataclass(frozen=True, slots=True)
class ProcessPlugin:
    """A reviewed process definition registered when the application starts."""

    name: str
    description: str
    state_model: type[BaseModel]
    initial_state: Callable[[], BaseModel]
    initial_step: str
    initial_prompt: str
    graph: CompiledStateGraph
    yields_to_suspended_reminders: bool = False

    def start(self) -> ProcessRecord:
        payload = self.initial_state().model_dump(mode="json")
        return self._record(
            status="waiting",
            step=self.initial_step,
            payload=payload,
            prompt=self.initial_prompt,
            result="",
        )

    def advance(self, record: ProcessRecord, message: str) -> ProcessRecord:
        if record.name != self.name:
            raise ValueError("Process record belongs to another plug-in")
        payload = self.state_model.model_validate(record.payload).model_dump(mode="json")
        result = self.graph.invoke(
            {
                **record.model_dump(mode="json"),
                "payload": payload,
                "message": message[:4_000],
            }
        )
        return self._record(
            status=result["status"],
            step=result["step"],
            payload=result["payload"],
            prompt=result.get("prompt", ""),
            result=result.get("result", ""),
        )

    def _record(
        self,
        *,
        status: ProcessStatus,
        step: str,
        payload: dict[str, Any],
        prompt: str,
        result: str,
    ) -> ProcessRecord:
        validated_payload = self.state_model.model_validate(payload).model_dump(mode="json")
        return ProcessRecord(
            name=self.name,
            status=status,
            step=step,
            payload=validated_payload,
            prompt=prompt,
            result=result,
        )
