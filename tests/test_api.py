from pathlib import Path

import bcrypt
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from chatbot.api.app import create_app
from chatbot.auth.service import AuthService
from chatbot.config import Settings
from chatbot.graph.builder import GraphDependencies, build_graph
from chatbot.graph.schemas import RouteDecision
from chatbot.graph.service import ChatbotService
from chatbot.memory.repository import MemoryRepository
from chatbot.memory.schemas import MemoryCandidate
from chatbot.retrieval.models import KnowledgeSnippet
from chatbot.services.container import AppContainer
from chatbot.tools.registry import ToolRegistry


class FakeKnowledgeBase:
    def search(self, query: str, *, limit: int = 4) -> list[KnowledgeSnippet]:
        return [KnowledgeSnippet(content="Shared knowledge", source="knowledge/guide.md")][:limit]


class FakeGateway:
    def decide_route(self, message: str) -> RouteDecision:
        return RouteDecision(route="chat", tool_name=None, tool_input=None)

    def generate(self, messages: list[BaseMessage]) -> str:
        return "Hello from the graph"

    def extract_memory(self, message: str) -> MemoryCandidate:
        return MemoryCandidate(
            should_store=True,
            content="I prefer concise answers",
            category="preference",
            confidence=0.95,
            sensitive=False,
        )


def make_container(tmp_path: Path) -> AppContainer:
    password_hash = bcrypt.hashpw(b"fixed-password", bcrypt.gensalt()).decode()
    settings = Settings(
        _env_file=None,
        openai_api_key=SecretStr("test-openai-key"),
        openai_model="test-model",
        auth_username="admin",
        auth_password_hash=SecretStr(password_hash),
        jwt_secret=SecretStr("a-test-jwt-secret-that-is-at-least-32-bytes"),
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        chroma_persist_directory=tmp_path / "chroma",
        knowledge_directory=tmp_path,
    )
    memories = MemoryRepository(settings.database_url)
    knowledge = FakeKnowledgeBase()
    graph = build_graph(
        GraphDependencies(
            model=FakeGateway(),
            knowledge_base=knowledge,
            tools=ToolRegistry(knowledge_base=knowledge),
            memories=memories,
        ),
        checkpointer=InMemorySaver(),
    )
    return AppContainer(
        settings=settings,
        auth=AuthService(
            username=settings.auth_username,
            password_hash=settings.auth_password_hash.get_secret_value(),
            jwt_secret=settings.jwt_secret.get_secret_value(),
        ),
        chatbot=ChatbotService(graph),
        memories=memories,
        knowledge_base=knowledge,
    )


def authenticate(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/auth/token",
        json={"username": "admin", "password": "fixed-password"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_auth_chat_and_health_contracts(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}
        assert (
            client.post("/auth/token", json={"username": "admin", "password": "wrong"}).status_code
            == 401
        )
        headers = authenticate(client)

        response = client.post(
            "/chat",
            headers=headers,
            json={"session_id": "demo-session", "message": "Hello"},
        )

        assert response.status_code == 200
        assert response.json()["response"] == "Hello from the graph"
        assert response.json()["route"] == "chat"
        assert response.json()["session_id"] == "demo-session"
    container.close()


def test_memory_endpoints_are_authenticated_and_owner_scoped(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        assert client.get("/memories").status_code == 401
        headers = authenticate(client)
        client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "memory-session",
                "message": "Remember my preference",
                "memory_consent": True,
            },
        )

        memories = client.get("/memories", headers=headers)
        assert memories.status_code == 200
        assert len(memories.json()) == 1
        memory_id = memories.json()[0]["id"]

        assert client.delete(f"/memories/{memory_id}", headers=headers).status_code == 204
        assert client.get("/memories", headers=headers).json() == []
    container.close()


def test_correlation_id_is_validated_before_logging_or_echoing(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        invalid = client.get("/health/live", headers={"X-Request-ID": "bad\r\nvalue"})
        valid = client.get("/health/live", headers={"X-Request-ID": "request-123"})

        assert invalid.headers["X-Request-ID"] != "bad\r\nvalue"
        assert valid.headers["X-Request-ID"] == "request-123"
    container.close()
