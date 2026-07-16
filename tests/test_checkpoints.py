from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from chatbot.services.checkpoints import CheckpointerResource


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
