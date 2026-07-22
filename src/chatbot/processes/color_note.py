"""Sequential process that stores and summarizes three model-extracted colors."""

from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from chatbot.processes.schemas import (
    ColorExtraction,
    ProcessGraphState,
    ProcessModel,
    ProcessPlugin,
)

_START_PROMPT = "Please input 3 colors."
_ColorName = Annotated[str, Field(min_length=1, max_length=40)]


class ColorNoteState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    colors: list[_ColorName] = Field(default_factory=list, max_length=3)


def build_color_note_plugin(model: ProcessModel) -> ProcessPlugin:
    builder = StateGraph(ProcessGraphState)
    builder.add_node("collect_colors", _collect_colors(model))
    builder.add_edge(START, "collect_colors")
    builder.add_edge("collect_colors", END)
    return ProcessPlugin(
        name="color_note",
        description=(
            "Add, remove, or edit up to three unique colors from user text and summarize them."
        ),
        state_model=ColorNoteState,
        initial_state=ColorNoteState,
        initial_step="collect_colors",
        initial_prompt=_START_PROMPT,
        graph=builder.compile(checkpointer=False),
    )


def _collect_colors(model: ProcessModel):
    def collect(state: ProcessGraphState) -> dict[str, object]:
        payload = ColorNoteState.model_validate(state.get("payload", {}))
        message = str(state.get("message", ""))[:4_000]
        extraction = ColorExtraction.model_validate(model.extract_colors(message))
        colors = list(payload.colors)
        for action in extraction.actions:
            color = _normalize(action.value)
            if action.action == "add":
                if color not in colors and len(colors) < 3:
                    colors.append(color)
            elif action.action == "remove":
                if color in colors:
                    colors.remove(color)
            elif color in colors and action.replacement is not None:
                replacement = _normalize(action.replacement)
                index = colors.index(color)
                if replacement in colors and replacement != color:
                    colors.pop(index)
                else:
                    colors[index] = replacement
        updated = ColorNoteState(colors=colors)
        if len(colors) == 3:
            return {
                "status": "completed",
                "step": "collect_colors",
                "payload": updated.model_dump(mode="json"),
                "prompt": "",
                "result": "Collected colors: " + ", ".join(colors) + ".",
            }
        if not colors:
            return {
                "status": "waiting",
                "step": "collect_colors",
                "payload": updated.model_dump(mode="json"),
                "prompt": _START_PROMPT,
                "result": "",
            }
        remaining = 3 - len(colors)
        unit = "color" if remaining == 1 else "colors"
        return {
            "status": "waiting",
            "step": "collect_colors",
            "payload": updated.model_dump(mode="json"),
            "prompt": f"Please input {remaining} more {unit}.",
            "result": "",
        }

    return collect


def _normalize(color: str) -> str:
    return " ".join(color.casefold().split())
