"""Small retrieval interfaces shared by the graph and tools."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class KnowledgeSnippet:
    content: str
    source: str


class KnowledgeBase(Protocol):
    def search(self, query: str, *, limit: int = 4) -> list[KnowledgeSnippet]: ...
