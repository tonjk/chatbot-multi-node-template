"""Strict public API request and response contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from chatbot.processes.schemas import ProcessAction, ProcessStatus


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenRequest(ApiModel):
    username: str = Field(min_length=1, max_length=128)
    password: SecretStr = Field(min_length=1, max_length=256)


class TokenResponse(ApiModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class ChatRequest(ApiModel):
    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    message: str = Field(min_length=1, max_length=4_000)
    memory_consent: bool = False
    process_action: ProcessAction = "auto"
    process_name: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_process_control(self) -> "ChatRequest":
        if self.process_action in {"start", "continue", "switch", "cancel"}:
            if self.process_name is None:
                raise ValueError("This process action requires process_name")
        elif self.process_action == "auto" and self.process_name is not None:
            raise ValueError("process_name requires an explicit process action")
        return self


class ProcessResponse(ApiModel):
    name: str
    status: ProcessStatus
    step: str
    active: bool


class ChatResponse(ApiModel):
    session_id: str
    response: str
    route: Literal["chat", "retrieve", "tools", "process"]
    correlation_id: str
    processes: list[ProcessResponse]


class MemoryResponse(ApiModel):
    id: str
    session_id: str
    content: str
    category: Literal["fact", "preference"]
    confidence: float
    created_at: datetime


class HealthResponse(ApiModel):
    status: Literal["ok", "ready"]
