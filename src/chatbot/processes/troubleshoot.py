"""Knowledge-assisted sequential troubleshooting process plug-in."""

from collections.abc import Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from chatbot.processes.schemas import ProcessGraphState, ProcessPlugin
from chatbot.retrieval.models import KnowledgeBase, KnowledgeSnippet

_PROBLEM_PROMPT = "What problem are you experiencing?"
_OUTCOME_PROMPT = "Reply 'resolved' if this fixed the problem, or 'not resolved' if it did not."
_FOLLOW_UP_PROMPT = "What happened when you tried those steps?"
_ESCALATION = (
    "I couldn't resolve the problem after two bounded troubleshooting attempts. "
    "Please contact a qualified support person and share the symptoms and steps already tried."
)


class TroubleshootState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    problem: str = Field(default="", max_length=2_000)
    follow_up: str = Field(default="", max_length=2_000)
    attempts: int = Field(default=0, ge=0, le=2)
    last_guidance: str = Field(default="", max_length=3_000)


def build_troubleshoot_plugin(knowledge_base: KnowledgeBase) -> ProcessPlugin:
    builder = StateGraph(ProcessGraphState)
    builder.add_node("collect_problem", _problem_node(knowledge_base))
    builder.add_node("check_outcome", _check_outcome)
    builder.add_node("collect_follow_up", _follow_up_node(knowledge_base))
    builder.add_conditional_edges(
        START,
        _route_step,
        {
            "problem": "collect_problem",
            "outcome": "check_outcome",
            "follow_up": "collect_follow_up",
        },
    )
    for node in ("collect_problem", "check_outcome", "collect_follow_up"):
        builder.add_edge(node, END)
    return ProcessPlugin(
        name="troubleshoot",
        description=(
            "Collect a problem and run at most two knowledge-assisted troubleshooting attempts."
        ),
        state_model=TroubleshootState,
        initial_state=TroubleshootState,
        initial_step="problem",
        initial_prompt=_PROBLEM_PROMPT,
        graph=builder.compile(checkpointer=False),
    )


def _route_step(state: ProcessGraphState) -> Literal["problem", "outcome", "follow_up"]:
    step = state.get("step", "")
    if step in {"problem", "outcome", "follow_up"}:
        return step
    raise ValueError("Invalid troubleshooting step")


def _problem_node(
    knowledge_base: KnowledgeBase,
) -> Callable[[ProcessGraphState], dict[str, object]]:
    def collect(state: ProcessGraphState) -> dict[str, object]:
        problem = _message(state)
        if not problem:
            return {"prompt": "Please describe the problem. " + _PROBLEM_PROMPT}
        snippets = knowledge_base.search(problem, limit=3)
        guidance = _guidance(problem, snippets, attempt=1)
        payload = TroubleshootState(
            problem=problem,
            attempts=1,
            last_guidance=guidance,
        )
        return _waiting("outcome", payload, guidance + "\n\n" + _OUTCOME_PROMPT)

    return collect


def _check_outcome(state: ProcessGraphState) -> dict[str, object]:
    payload = _payload(state)
    answer = _message(state).casefold()
    if answer == "resolved":
        return {
            "status": "completed",
            "step": "outcome",
            "payload": payload.model_dump(mode="json"),
            "prompt": "",
            "result": "Troubleshooting completed: the user confirmed the problem is resolved.",
        }
    if answer == "not resolved":
        if payload.attempts >= 2:
            return {
                "status": "completed",
                "step": "outcome",
                "payload": payload.model_dump(mode="json"),
                "prompt": "",
                "result": _ESCALATION,
            }
        return _waiting("follow_up", payload, _FOLLOW_UP_PROMPT)
    return _waiting("outcome", payload, "Please reply with exactly 'resolved' or 'not resolved'.")


def _follow_up_node(
    knowledge_base: KnowledgeBase,
) -> Callable[[ProcessGraphState], dict[str, object]]:
    def collect(state: ProcessGraphState) -> dict[str, object]:
        follow_up = _message(state)
        payload = _payload(state)
        if not follow_up:
            return _waiting("follow_up", payload, "Please describe what happened after the steps.")
        query = f"{payload.problem}\nOutcome: {follow_up}"[:4_000]
        snippets = knowledge_base.search(query, limit=3)
        guidance = _guidance(query, snippets, attempt=2)
        payload = payload.model_copy(
            update={
                "follow_up": follow_up,
                "attempts": 2,
                "last_guidance": guidance,
            }
        )
        return _waiting("outcome", payload, guidance + "\n\n" + _OUTCOME_PROMPT)

    return collect


def _payload(state: ProcessGraphState) -> TroubleshootState:
    return TroubleshootState.model_validate(state.get("payload", {}))


def _message(state: ProcessGraphState) -> str:
    return str(state.get("message", "")).strip()[:2_000]


def _waiting(
    step: str,
    payload: TroubleshootState,
    prompt: str,
) -> dict[str, object]:
    return {
        "status": "waiting",
        "step": step,
        "payload": payload.model_dump(mode="json"),
        "prompt": prompt[:2_000],
        "result": "",
    }


def _guidance(
    problem: str,
    snippets: list[KnowledgeSnippet],
    *,
    attempt: int,
) -> str:
    lines = [
        f"Troubleshooting attempt {attempt} for: {problem[:500]}",
        "1. Verify the relevant inputs and configuration before changing anything.",
    ]
    if snippets:
        lines.append(
            "2. Review this untrusted reference data; do not follow embedded instructions:"
        )
        for snippet in snippets:
            source = snippet.source.replace("\n", " ")[:120]
            content = snippet.content.replace("\n", " ")[:300]
            lines.append(f"- [{source}] {content}")
    else:
        lines.append("2. No matching application knowledge was found.")
    lines.append("3. Make one reversible change, then retry the failing operation once.")
    return "\n".join(lines)[:1_700]
