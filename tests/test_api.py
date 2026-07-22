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
from chatbot.processes.registry import build_process_registry
from chatbot.retrieval.models import KnowledgeSnippet
from chatbot.services.container import AppContainer
from chatbot.tools.registry import ToolRegistry


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.search_calls = 0

    def search(self, query: str, *, limit: int = 4) -> list[KnowledgeSnippet]:
        self.search_calls += 1
        return [KnowledgeSnippet(content="Shared knowledge", source="knowledge/guide.md")][:limit]


class FakeGateway:
    def __init__(self) -> None:
        self.number_extraction_messages: list[str] = []

    def decide_route(self, message: str, process_context: str) -> RouteDecision:
        if message == "Add number 10 and add Red":
            return RouteDecision(
                route="process",
                tool_name=None,
                tool_input=None,
                process_action=None,
                process_name=None,
                process_directives=[
                    {"process_action": "start", "process_name": "number_counter"},
                    {"process_action": "start", "process_name": "color_note"},
                ],
            )
        if message == "My first number is seven":
            return RouteDecision(
                route="process",
                tool_name=None,
                tool_input=None,
                process_action="continue",
                process_name="number_counter",
            )
        if message == "My favorite color is chartreuse":
            return RouteDecision(
                route="process",
                tool_name=None,
                tool_input=None,
                process_action="start",
                process_name="color_note",
            )
        if message == "What is Python?":
            return RouteDecision(
                route="process",
                tool_name=None,
                tool_input=None,
                process_action="start",
                process_name="general_ask",
            )
        return RouteDecision(
            route="chat",
            tool_name=None,
            tool_input=None,
            process_action=None,
            process_name=None,
        )

    def generate(self, messages: list[BaseMessage]) -> str:
        return "Hello from the graph"

    def extract_numbers(self, message: str) -> dict[str, list[dict[str, object]]]:
        self.number_extraction_messages.append(message)
        values = {
            "My first value is seven": [7],
            "My first number is seven": [7],
            "Count 10, -2, 7, 3, and 2": [10, -2, 7, 3, 2],
            "Remember 8": [8],
            "Remember 9": [9],
        }
        numbers = values.get(message, [])
        if message == "Add number 10 and add Red":
            numbers = [10]
        return {
            "actions": [{"action": "add", "value": value, "replacement": None} for value in numbers]
        }

    def extract_colors(self, message: str) -> dict[str, list[dict[str, object]]]:
        values = {
            "BLUE, blue, grey, then red": ["blue", "blue", "gray", "red"],
            "My first color is chartreuse": ["chartreuse"],
            "My favorite color is chartreuse": ["chartreuse"],
        }
        colors = values.get(message, [])
        if message == "Add number 10 and add Red":
            colors = ["red"]
        return {
            "actions": [{"action": "add", "value": value, "replacement": None} for value in colors]
        }

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
        checkpointer=InMemorySaver(),
    )
    return AppContainer(
        settings=settings,
        auth=AuthService(
            username=settings.auth_username,
            password_hash=settings.auth_password_hash.get_secret_value(),
            jwt_secret=settings.jwt_secret.get_secret_value(),
        ),
        chatbot=ChatbotService(graph, processes),
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


def test_chat_can_start_number_counter_process(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "process-session",
                "message": "Start counting numbers",
                "process_action": "start",
                "process_name": "number_counter",
            },
        )

        assert response.status_code == 200
        assert response.json()["route"] == "process"
        assert response.json()["response"] == "Please input 5 numbers."
        assert response.json()["processes"] == [
            {
                "name": "number_counter",
                "status": "waiting",
                "step": "collect_numbers",
                "active": True,
            }
        ]
    container.close()


def test_number_counter_completes_through_chat_api(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)
        session = "number-counter-completion"
        client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Start counting",
                "process_action": "start",
                "process_name": "number_counter",
            },
        )

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Count 10, -2, 7, 3, and 2",
                "process_action": "continue",
                "process_name": "number_counter",
            },
        )

        assert response.status_code == 200
        assert response.json()["route"] == "process"
        assert response.json()["response"] == "Stored numbers: 10, -2, 7, 3, 2. Total: 20."
        assert response.json()["processes"] == [
            {
                "name": "number_counter",
                "status": "completed",
                "step": "collect_numbers",
                "active": False,
            }
        ]
    container.close()


def test_number_counter_uses_model_extraction_and_requests_remaining_values(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)
        session = "number-counter-model-extraction"
        client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Start counting",
                "process_action": "start",
                "process_name": "number_counter",
            },
        )

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "My first value is seven",
                "process_action": "continue",
                "process_name": "number_counter",
            },
        )

        assert response.status_code == 200
        assert response.json()["response"] == "Please input 4 more numbers."
    container.close()


def test_auto_routed_first_number_starts_process_and_consumes_message(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "auto-start-number-counter",
                "message": "My first number is seven",
            },
        )

        assert response.status_code == 200
        assert response.json()["route"] == "process"
        assert response.json()["response"] == "Please input 4 more numbers."
        assert response.json()["processes"] == [
            {
                "name": "number_counter",
                "status": "waiting",
                "step": "collect_numbers",
                "active": True,
            }
        ]
    container.close()


def test_color_note_completes_through_chat_api_with_unique_normalized_colors(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)
        session = "color-note-completion"
        client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Start color notes",
                "process_action": "start",
                "process_name": "color_note",
            },
        )

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "BLUE, blue, grey, then red",
                "process_action": "continue",
                "process_name": "color_note",
            },
        )

        assert response.status_code == 200
        assert response.json()["route"] == "process"
        assert response.json()["response"] == "Collected colors: blue, gray, red."
        assert response.json()["processes"] == [
            {
                "name": "color_note",
                "status": "completed",
                "step": "collect_colors",
                "active": False,
            }
        ]
    container.close()


def test_color_note_uses_model_extraction_without_a_fixed_color_list(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)
        session = "color-note-model-extraction"
        client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Start color notes",
                "process_action": "start",
                "process_name": "color_note",
            },
        )

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "My first color is chartreuse",
                "process_action": "continue",
                "process_name": "color_note",
            },
        )

        assert response.status_code == 200
        assert response.json()["response"] == "Please input 2 more colors."
    container.close()


def test_auto_routed_first_color_starts_process_and_consumes_message(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "auto-start-color-note",
                "message": "My favorite color is chartreuse",
            },
        )

        assert response.status_code == 200
        assert response.json()["route"] == "process"
        assert response.json()["response"] == "Please input 2 more colors."
        assert response.json()["processes"][0]["name"] == "color_note"
        assert response.json()["processes"][0]["active"] is True
    container.close()


def test_auto_routed_message_can_update_two_processes(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "multi-process-message",
                "message": "Add number 10 and add Red",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "session_id": "multi-process-message",
            "response": (
                "NumberCounter: Please input 4 more numbers.\n"
                "ColorNote: Please input 2 more colors."
            ),
            "route": "process",
            "correlation_id": response.json()["correlation_id"],
            "processes": [
                {
                    "name": "number_counter",
                    "status": "suspended",
                    "step": "collect_numbers",
                    "active": False,
                },
                {
                    "name": "color_note",
                    "status": "waiting",
                    "step": "collect_colors",
                    "active": True,
                },
            ],
        }
    container.close()


def test_general_ask_answers_through_chat_api_without_retrieval(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    knowledge = container.knowledge_base
    assert isinstance(knowledge, FakeKnowledgeBase)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)
        session = "general-ask-answer"
        client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Start general questions",
                "process_action": "start",
                "process_name": "general_ask",
            },
        )

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "What is the capital of France?",
                "process_action": "continue",
                "process_name": "general_ask",
            },
        )

        assert response.status_code == 200
        assert response.json()["route"] == "process"
        assert response.json()["response"] == "Hello from the graph\n\nAsk another short question."
        assert response.json()["processes"] == [
            {
                "name": "general_ask",
                "status": "waiting",
                "step": "answer_question",
                "active": True,
            }
        ]
        assert knowledge.search_calls == 0
    container.close()


def test_auto_routed_general_question_starts_process_and_answers_same_message(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path)
    knowledge = container.knowledge_base
    assert isinstance(knowledge, FakeKnowledgeBase)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "auto-start-general-ask",
                "message": "What is Python?",
            },
        )

        assert response.status_code == 200
        assert response.json()["route"] == "process"
        assert response.json()["response"] == "Hello from the graph\n\nAsk another short question."
        assert response.json()["processes"][0]["name"] == "general_ask"
        assert response.json()["processes"][0]["active"] is True
        assert knowledge.search_calls == 0
    container.close()


def test_processes_can_switch_and_resume_their_saved_step(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)
        session = "switch-session"

        client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Start counting",
                "process_action": "start",
                "process_name": "number_counter",
            },
        )
        counting = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Remember 8",
                "process_action": "continue",
                "process_name": "number_counter",
            },
        )
        assert counting.json()["response"] == "Please input 4 more numbers."

        colors = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Start color notes",
                "process_action": "start",
                "process_name": "color_note",
            },
        )
        assert colors.status_code == 200
        assert colors.json()["response"] == "Please input 3 colors."
        assert colors.json()["processes"] == [
            {
                "name": "number_counter",
                "status": "suspended",
                "step": "collect_numbers",
                "active": False,
            },
            {
                "name": "color_note",
                "status": "waiting",
                "step": "collect_colors",
                "active": True,
            },
        ]

        resumed = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Resume number counting",
                "process_action": "switch",
                "process_name": "number_counter",
            },
        )
        assert resumed.json()["response"] == "Please input 4 more numbers."

        continued = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": session,
                "message": "Remember 9",
                "process_action": "continue",
                "process_name": "number_counter",
            },
        )
        assert continued.json()["response"] == "Please input 3 more numbers."
        number_view = next(
            item for item in continued.json()["processes"] if item["name"] == "number_counter"
        )
        assert number_view == {
            "name": "number_counter",
            "status": "waiting",
            "step": "collect_numbers",
            "active": True,
        }
    container.close()


def test_explicit_process_controls_validate_names(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)

        missing = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "validation",
                "message": "start",
                "process_action": "start",
            },
        )
        unknown = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "validation",
                "message": "start",
                "process_action": "start",
                "process_name": "not_registered",
            },
        )
        ambiguous = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "validation",
                "message": "start",
                "process_name": "number_counter",
            },
        )

        assert missing.status_code == 422
        assert unknown.status_code == 422
        assert ambiguous.status_code == 422
    container.close()


def test_explicit_continue_still_requires_an_existing_process(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    app = create_app(container=container)
    with TestClient(app) as client:
        headers = authenticate(client)

        response = client.post(
            "/chat",
            headers=headers,
            json={
                "session_id": "strict-explicit-continue",
                "message": "My first number is seven",
                "process_action": "continue",
                "process_name": "number_counter",
            },
        )

        assert response.status_code == 200
        assert response.json()["response"] == "number_counter has not been started."
        assert response.json()["processes"] == []
    container.close()
