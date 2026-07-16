"""Validated durable-memory contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryCandidate(BaseModel):
    """A model-proposed fact or preference that may be safe to retain."""

    model_config = ConfigDict(extra="forbid", strict=True)

    should_store: bool
    content: str = Field(min_length=1, max_length=500)
    category: Literal["fact", "preference"]
    confidence: float = Field(ge=0, le=1)
    sensitive: bool


class MemoryRecord(BaseModel):
    """A durable memory returned through the authenticated API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    session_id: str
    content: str
    category: Literal["fact", "preference"]
    confidence: float
    created_at: datetime
