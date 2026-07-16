from datetime import UTC, datetime

import pytest

from chatbot.retrieval.models import KnowledgeSnippet
from chatbot.tools.registry import ToolRegistry, ToolValidationError


class FakeKnowledgeBase:
    def search(self, query: str, *, limit: int = 4) -> list[KnowledgeSnippet]:
        return [
            KnowledgeSnippet(content=f"Result for {query}", source="knowledge/guide.md"),
        ][:limit]


def registry() -> ToolRegistry:
    return ToolRegistry(
        knowledge_base=FakeKnowledgeBase(),
        clock=lambda: datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )


def test_calculator_evaluates_only_bounded_arithmetic() -> None:
    tools = registry()

    result = tools.invoke("calculator", {"expression": "2 + 3 * 4"})

    assert result.content == "14"
    with pytest.raises(ToolValidationError):
        tools.invoke("calculator", {"expression": "__import__('os').system('id')"})
    with pytest.raises(ToolValidationError):
        tools.invoke("calculator", {"expression": "2 ** 100"})


def test_registry_rejects_unknown_tools_and_extra_arguments() -> None:
    tools = registry()

    with pytest.raises(ToolValidationError, match="not available"):
        tools.invoke("shell", {"command": "id"})
    with pytest.raises(ToolValidationError, match="Invalid tool input"):
        tools.invoke("calculator", {"expression": "1 + 1", "extra": "ignored"})


def test_current_time_validates_timezone() -> None:
    tools = registry()

    result = tools.invoke("current_time", {"timezone": "Asia/Bangkok"})

    assert result.content == "2026-01-01T19:00:00+07:00"
    with pytest.raises(ToolValidationError, match="Unknown timezone"):
        tools.invoke("current_time", {"timezone": "Mars/Olympus"})


def test_knowledge_search_uses_the_shared_bounded_index() -> None:
    tools = registry()

    result = tools.invoke("knowledge_search", {"query": "memory consent"})

    assert result.content == "[knowledge/guide.md]\nResult for memory consent"
