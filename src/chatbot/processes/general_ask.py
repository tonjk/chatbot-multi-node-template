"""Short, model-only general question process with no retrieval."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from chatbot.processes.schemas import ProcessGraphState, ProcessModel, ProcessPlugin

_START_PROMPT = (
    "What would you like to ask? I will answer briefly using the AI model's general knowledge."
)
_SYSTEM_PROMPT = (
    "Answer the user's general-knowledge question directly in at most two short sentences. "
    "Do not claim to have searched a knowledge base, retrieved documents, or used external tools."
)


class GeneralAskState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    questions_asked: int = Field(default=0, ge=0, le=1_000)


def build_general_ask_plugin(model: ProcessModel) -> ProcessPlugin:
    builder = StateGraph(ProcessGraphState)
    builder.add_node("answer_question", _answer_node(model))
    builder.add_edge(START, "answer_question")
    builder.add_edge("answer_question", END)
    return ProcessPlugin(
        name="general_ask",
        description="Answer short general-knowledge questions using only the configured AI model.",
        state_model=GeneralAskState,
        initial_state=GeneralAskState,
        initial_step="answer_question",
        initial_prompt=_START_PROMPT,
        graph=builder.compile(checkpointer=False),
        yields_to_suspended_reminders=True,
    )


def _answer_node(model: ProcessModel):
    def answer(state: ProcessGraphState) -> dict[str, object]:
        payload = GeneralAskState.model_validate(state.get("payload", {}))
        question = str(state.get("message", "")).strip()[:4_000]
        if not question:
            return {
                "status": "waiting",
                "step": "answer_question",
                "payload": payload.model_dump(mode="json"),
                "prompt": "Please ask a non-blank general-knowledge question.",
                "result": "",
            }
        response = model.generate(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=question)]
        ).strip()
        if not response:
            raise ValueError("Model returned an empty general answer")
        updated = payload.model_copy(update={"questions_asked": payload.questions_asked + 1})
        return {
            "status": "waiting",
            "step": "answer_question",
            "payload": updated.model_dump(mode="json"),
            "prompt": response[:600] + "\n\nAsk another short question.",
            "result": "",
        }

    return answer
