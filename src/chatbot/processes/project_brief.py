"""Sequential project-brief process plug-in."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from chatbot.processes.schemas import ProcessGraphState, ProcessPlugin

_GOAL_PROMPT = "What goal should this project accomplish?"
_AUDIENCE_PROMPT = "Who is the intended audience for this project?"
_CONSTRAINTS_PROMPT = "What constraints, requirements, or limits should the project respect?"
_CONFIRM_PROMPT = "Reply 'approve' to finish, or 'revise' to change the brief."
_REVISION_PROMPT = "What changes should I apply to the project brief?"


class ProjectBriefState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    goal: str = Field(default="", max_length=2_000)
    audience: str = Field(default="", max_length=2_000)
    constraints: str = Field(default="", max_length=2_000)
    revisions: list[str] = Field(default_factory=list, max_length=3)
    brief: str = Field(default="", max_length=4_000)


def build_project_brief_plugin() -> ProcessPlugin:
    builder = StateGraph(ProcessGraphState)
    builder.add_node("collect_goal", _collect_goal)
    builder.add_node("collect_audience", _collect_audience)
    builder.add_node("collect_constraints", _collect_constraints)
    builder.add_node("confirm", _confirm)
    builder.add_node("revise", _revise)
    builder.add_conditional_edges(
        START,
        _route_step,
        {
            "goal": "collect_goal",
            "audience": "collect_audience",
            "constraints": "collect_constraints",
            "confirmation": "confirm",
            "revision": "revise",
        },
    )
    for node in ("collect_goal", "collect_audience", "collect_constraints", "confirm", "revise"):
        builder.add_edge(node, END)
    return ProcessPlugin(
        name="project_brief",
        description="Collect a goal, audience, and constraints, then review a project brief.",
        state_model=ProjectBriefState,
        initial_state=ProjectBriefState,
        initial_step="goal",
        initial_prompt=_GOAL_PROMPT,
        graph=builder.compile(checkpointer=False),
    )


def _route_step(
    state: ProcessGraphState,
) -> Literal["goal", "audience", "constraints", "confirmation", "revision"]:
    step = state.get("step", "")
    if step in {"goal", "audience", "constraints", "confirmation", "revision"}:
        return step
    raise ValueError("Invalid project-brief step")


def _collect_goal(state: ProcessGraphState) -> dict[str, object]:
    value = _required_text(state, _GOAL_PROMPT)
    if value is None:
        return {"prompt": "Please provide a non-blank goal. " + _GOAL_PROMPT}
    payload = _payload(state).model_copy(update={"goal": value})
    return _waiting("audience", payload, _AUDIENCE_PROMPT)


def _collect_audience(state: ProcessGraphState) -> dict[str, object]:
    value = _required_text(state, _AUDIENCE_PROMPT)
    if value is None:
        return {"prompt": "Please provide a non-blank audience. " + _AUDIENCE_PROMPT}
    payload = _payload(state).model_copy(update={"audience": value})
    return _waiting("constraints", payload, _CONSTRAINTS_PROMPT)


def _collect_constraints(state: ProcessGraphState) -> dict[str, object]:
    value = _required_text(state, _CONSTRAINTS_PROMPT)
    if value is None:
        return {"prompt": "Please provide at least one constraint or write 'none'."}
    payload = _payload(state).model_copy(update={"constraints": value})
    payload = payload.model_copy(update={"brief": _render_brief(payload)})
    return _waiting("confirmation", payload, payload.brief + "\n\n" + _CONFIRM_PROMPT)


def _confirm(state: ProcessGraphState) -> dict[str, object]:
    payload = _payload(state)
    answer = str(state.get("message", "")).strip().casefold()
    if answer == "approve":
        return {
            "status": "completed",
            "step": "confirmation",
            "payload": payload.model_dump(mode="json"),
            "prompt": "",
            "result": payload.brief,
        }
    if answer == "revise":
        if len(payload.revisions) >= 3:
            return _waiting(
                "confirmation",
                payload,
                "The three-revision limit has been reached. Reply 'approve' to finish.",
            )
        return _waiting("revision", payload, _REVISION_PROMPT)
    return _waiting(
        "confirmation",
        payload,
        "Please reply with exactly 'approve' or 'revise'.",
    )


def _revise(state: ProcessGraphState) -> dict[str, object]:
    value = _required_text(state, _REVISION_PROMPT)
    payload = _payload(state)
    if value is None:
        return _waiting("revision", payload, "Please describe the requested changes.")
    revisions = [*payload.revisions, value]
    payload = payload.model_copy(update={"revisions": revisions})
    payload = payload.model_copy(update={"brief": _render_brief(payload)})
    return _waiting("confirmation", payload, payload.brief + "\n\n" + _CONFIRM_PROMPT)


def _payload(state: ProcessGraphState) -> ProjectBriefState:
    return ProjectBriefState.model_validate(state.get("payload", {}))


def _required_text(state: ProcessGraphState, _prompt: str) -> str | None:
    value = str(state.get("message", "")).strip()
    return value[:2_000] if value else None


def _waiting(
    step: str,
    payload: ProjectBriefState,
    prompt: str,
) -> dict[str, object]:
    return {
        "status": "waiting",
        "step": step,
        "payload": payload.model_dump(mode="json"),
        "prompt": prompt[:2_000],
        "result": "",
    }


def _render_brief(payload: ProjectBriefState) -> str:
    lines = [
        "Project brief",
        f"Goal: {payload.goal}",
        f"Audience: {payload.audience}",
        f"Constraints: {payload.constraints}",
    ]
    if payload.revisions:
        lines.append("Revisions: " + "; ".join(payload.revisions))
    return "\n".join(lines)[:4_000]
