"""Sequential process that stores five model-extracted integers and reports their sum."""

from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from chatbot.processes.schemas import (
    NumberExtraction,
    ProcessGraphState,
    ProcessModel,
    ProcessPlugin,
)

_START_PROMPT = "Please input 5 numbers."
_BoundedInteger = Annotated[int, Field(ge=-1_000_000_000, le=1_000_000_000)]


class NumberCounterState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    numbers: list[_BoundedInteger] = Field(default_factory=list, max_length=5)


def build_number_counter_plugin(model: ProcessModel) -> ProcessPlugin:
    builder = StateGraph(ProcessGraphState)
    builder.add_node("collect_numbers", _collect_numbers(model))
    builder.add_edge(START, "collect_numbers")
    builder.add_edge("collect_numbers", END)
    return ProcessPlugin(
        name="number_counter",
        description=(
            "Add, remove, or edit up to five integer values from user text and report their sum."
        ),
        state_model=NumberCounterState,
        initial_state=NumberCounterState,
        initial_step="collect_numbers",
        initial_prompt=_START_PROMPT,
        graph=builder.compile(checkpointer=False),
    )


def _collect_numbers(model: ProcessModel):
    def collect(state: ProcessGraphState) -> dict[str, object]:
        payload = NumberCounterState.model_validate(state.get("payload", {}))
        message = str(state.get("message", ""))[:4_000]
        extraction = NumberExtraction.model_validate(model.extract_numbers(message))
        numbers = list(payload.numbers)
        for action in extraction.actions:
            if action.action == "add":
                if len(numbers) < 5:
                    numbers.append(action.value)
            elif action.action == "remove":
                if action.value in numbers:
                    numbers.remove(action.value)
            elif action.value in numbers and action.replacement is not None:
                numbers[numbers.index(action.value)] = action.replacement
        payload = NumberCounterState(numbers=numbers)
        if len(numbers) == 5:
            rendered = ", ".join(str(value) for value in numbers)
            return {
                "status": "completed",
                "step": "collect_numbers",
                "payload": payload.model_dump(mode="json"),
                "prompt": "",
                "result": f"Stored numbers: {rendered}. Total: {sum(numbers)}.",
            }

        if not numbers:
            return _waiting(payload, _START_PROMPT)
        remaining = 5 - len(numbers)
        unit = "number" if remaining == 1 else "numbers"
        return _waiting(payload, f"Please input {remaining} more {unit}.")

    return collect


def _waiting(payload: NumberCounterState, prompt: str) -> dict[str, object]:
    return {
        "status": "waiting",
        "step": "collect_numbers",
        "payload": payload.model_dump(mode="json"),
        "prompt": prompt,
        "result": "",
    }
