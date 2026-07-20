"""Assemble and compile the focused LangGraph workflow."""

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot.graph.schemas import ModelGateway
from chatbot.graph.state import ChatState, GraphContext
from chatbot.memory.repository import MemoryRepository
from chatbot.processes.registry import ProcessRegistry
from chatbot.retrieval.models import KnowledgeBase
from chatbot.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    model: ModelGateway
    knowledge_base: KnowledgeBase
    tools: ToolRegistry
    memories: MemoryRepository
    processes: ProcessRegistry


def build_graph(
    dependencies: GraphDependencies,
    *,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph:
    """Compile the graph once for reuse throughout the app lifespan."""

    from chatbot.graph.nodes import (
        ChatNodes,
        route_after_process_control,
        route_after_router,
    )

    nodes = ChatNodes(dependencies)
    builder = StateGraph(ChatState, context_schema=GraphContext)
    builder.add_node("router", nodes.router)
    builder.add_node("chat", nodes.chat)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("tools", nodes.tools)
    builder.add_node("process_control", nodes.process_control)
    builder.add_node("process_dispatch", nodes.process_dispatch)
    builder.add_node("memory", nodes.memory)
    builder.add_node("respond", nodes.respond)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "chat": "chat",
            "retrieve": "retrieve",
            "tools": "tools",
            "process": "process_control",
            "respond": "respond",
        },
    )
    builder.add_edge("chat", "memory")
    builder.add_edge("retrieve", "memory")
    builder.add_edge("tools", "memory")
    builder.add_conditional_edges(
        "process_control",
        route_after_process_control,
        {
            "dispatch": "process_dispatch",
            "respond": "respond",
        },
    )
    builder.add_edge("process_dispatch", "memory")
    builder.add_edge("memory", "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)
