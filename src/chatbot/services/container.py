"""Construct shared resources once for the FastAPI lifespan."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import timedelta

from langchain_openai import OpenAIEmbeddings

from chatbot.auth.service import AuthService
from chatbot.config import Settings
from chatbot.graph.builder import GraphDependencies, build_graph
from chatbot.graph.service import ChatbotService
from chatbot.memory.repository import MemoryRepository
from chatbot.processes.registry import build_process_registry
from chatbot.retrieval.chroma import ChromaKnowledgeBase
from chatbot.retrieval.models import KnowledgeBase
from chatbot.services.checkpoints import CheckpointerResource
from chatbot.services.openai_gateway import OpenAIModelGateway
from chatbot.tools.registry import ToolRegistry


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    auth: AuthService
    chatbot: ChatbotService
    memories: MemoryRepository
    knowledge_base: KnowledgeBase
    _closer: Callable[[], None] | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._closer is not None:
            self._closer()
        else:
            self.memories.close()


def build_container(settings: Settings) -> AppContainer:
    """Build provider, persistence, graph, and auth services with safe cleanup."""

    settings.require_openai()
    stack = ExitStack()
    try:
        memories = MemoryRepository(settings.database_url)
        stack.callback(memories.close)
        checkpointer = stack.enter_context(CheckpointerResource(settings.database_url))
        embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        knowledge_base = ChromaKnowledgeBase(
            persist_directory=settings.chroma_persist_directory,
            embeddings=embeddings,
        )
        model = OpenAIModelGateway(settings)
        tools = ToolRegistry(knowledge_base=knowledge_base)
        processes = build_process_registry(model)
        graph = build_graph(
            GraphDependencies(
                model=model,
                knowledge_base=knowledge_base,
                tools=tools,
                memories=memories,
                processes=processes,
            ),
            checkpointer=checkpointer,
        )
        auth = AuthService(
            username=settings.auth_username,
            password_hash=settings.auth_password_hash.get_secret_value(),
            jwt_secret=settings.jwt_secret.get_secret_value(),
            token_ttl=timedelta(minutes=30),
        )
        return AppContainer(
            settings=settings,
            auth=auth,
            chatbot=ChatbotService(graph, processes),
            memories=memories,
            knowledge_base=knowledge_base,
            _closer=stack.close,
        )
    except Exception:
        stack.close()
        raise
