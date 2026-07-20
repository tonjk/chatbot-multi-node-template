import re
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from chatbot.graph.builder import GraphDependencies, build_graph
from chatbot.graph.schemas import RouteDecision
from chatbot.graph.service import ChatbotService
from chatbot.memory.repository import MemoryRepository
from chatbot.memory.schemas import MemoryCandidate
from chatbot.processes.registry import build_process_registry
from chatbot.retrieval.models import KnowledgeSnippet
from chatbot.services.checkpoints import CheckpointerResource
from chatbot.tools.registry import ToolRegistry


class State(TypedDict):
    value: str


def build_test_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("save", lambda state: {"value": state["value"]})
    builder.add_edge(START, "save")
    builder.add_edge("save", END)
    return builder.compile(checkpointer=checkpointer)


def test_sqlite_checkpoints_survive_resource_reopen(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'checkpoints.db'}"
    config = {"configurable": {"thread_id": "subject-session-hash"}}

    with CheckpointerResource(database_url) as checkpointer:
        build_test_graph(checkpointer).invoke({"value": "persisted"}, config=config)

    with CheckpointerResource(database_url) as checkpointer:
        snapshot = build_test_graph(checkpointer).get_state(config)

    assert snapshot.values["value"] == "persisted"


class FakeKnowledgeBase:
    def search(self, query: str, *, limit: int = 4) -> list[KnowledgeSnippet]:
        return [KnowledgeSnippet(content="Reference", source="knowledge/test.md")][:limit]


class FakeGateway:
    def decide_route(
        self,
        message: str,
        process_context: str,
    ) -> RouteDecision | dict[str, Any]:
        return RouteDecision(
            route="chat",
            tool_name=None,
            tool_input=None,
            process_action=None,
            process_name=None,
        )

    def generate(self, messages: list[BaseMessage]) -> str:
        return "Generated"

    def extract_numbers(self, message: str) -> dict[str, list[int]]:
        numbers = re.findall(r"(?<![\w.])[+-]?\d+(?![\w.])", message)
        return {"numbers": [int(value) for value in numbers]}

    def extract_colors(self, message: str) -> dict[str, list[str]]:
        return {"colors": []}

    def extract_memory(self, message: str) -> MemoryCandidate:
        return MemoryCandidate(
            should_store=False,
            content="",
            category="fact",
            confidence=0.0,
            sensitive=False,
        )


def build_process_service(checkpointer, memories: MemoryRepository) -> ChatbotService:
    knowledge = FakeKnowledgeBase()
    model = FakeGateway()
    processes = build_process_registry(model)
    graph = build_graph(
        GraphDependencies(
            model=model,
            knowledge_base=knowledge,
            tools=ToolRegistry(knowledge_base=knowledge),
            memories=memories,
            processes=processes,
        ),
        checkpointer=checkpointer,
    )
    return ChatbotService(graph, processes)


def test_process_progress_survives_sqlite_checkpointer_reopen(tmp_path: Path) -> None:
    checkpoint_url = f"sqlite:///{tmp_path / 'process-checkpoints.db'}"
    memory_url = f"sqlite:///{tmp_path / 'process-memories.db'}"
    first_memories = MemoryRepository(memory_url)
    with CheckpointerResource(checkpoint_url) as checkpointer:
        service = build_process_service(checkpointer, first_memories)
        service.chat(
            subject="alice",
            session_id="durable",
            message="start",
            process_action="start",
            process_name="number_counter",
        )
        service.chat(
            subject="alice",
            session_id="durable",
            message="Remember 41",
            process_action="continue",
            process_name="number_counter",
        )
    first_memories.close()

    reopened_memories = MemoryRepository(memory_url)
    with CheckpointerResource(checkpoint_url) as checkpointer:
        service = build_process_service(checkpointer, reopened_memories)
        resumed = service.chat(
            subject="alice",
            session_id="durable",
            message="resume",
            process_action="switch",
            process_name="number_counter",
        )
    reopened_memories.close()

    assert resumed.response == "Please input 4 more numbers."
    assert resumed.processes[0].step == "collect_numbers"
    assert resumed.processes[0].active is True
