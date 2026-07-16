"""Validated allow-list for chatbot tools."""

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatbot.retrieval.models import KnowledgeBase

_MAX_OUTPUT_CHARS = 4_000
_MAX_ABSOLUTE_VALUE = 1_000_000_000_000


class ToolValidationError(ValueError):
    """Raised for a rejected tool name, input, or execution."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    content: str


class _CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expression: str = Field(min_length=1, max_length=128)


class _CurrentTimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    timezone: str = Field(default="UTC", min_length=1, max_length=64)


class _KnowledgeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=500)


class ToolRegistry:
    """Dispatch only explicitly registered tools after schema validation."""

    def __init__(
        self,
        *,
        knowledge_base: KnowledgeBase,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._clock = clock or (lambda: datetime.now(UTC))
        self._handlers: dict[str, Callable[[Mapping[str, Any]], ToolResult]] = {
            "calculator": self._calculator,
            "current_time": self._current_time,
            "knowledge_search": self._knowledge_search,
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolValidationError(f"Tool {name!r} is not available")
        try:
            result = handler(arguments)
        except ValidationError:
            raise ToolValidationError("Invalid tool input") from None
        except ToolValidationError:
            raise
        except Exception:
            raise ToolValidationError("Tool execution failed safely") from None
        return ToolResult(name=result.name, content=result.content[:_MAX_OUTPUT_CHARS])

    def _calculator(self, arguments: Mapping[str, Any]) -> ToolResult:
        parsed = _CalculatorInput.model_validate(arguments)
        result = _evaluate_arithmetic(parsed.expression)
        rendered = str(result) if isinstance(result, int) else format(result, ".12g")
        return ToolResult(name="calculator", content=rendered)

    def _current_time(self, arguments: Mapping[str, Any]) -> ToolResult:
        parsed = _CurrentTimeInput.model_validate(arguments)
        try:
            timezone = ZoneInfo(parsed.timezone)
        except ZoneInfoNotFoundError:
            raise ToolValidationError("Unknown timezone") from None
        now = self._clock()
        if now.tzinfo is None:
            raise ToolValidationError("Clock returned an invalid time")
        return ToolResult(name="current_time", content=now.astimezone(timezone).isoformat())

    def _knowledge_search(self, arguments: Mapping[str, Any]) -> ToolResult:
        parsed = _KnowledgeSearchInput.model_validate(arguments)
        snippets = self._knowledge_base.search(parsed.query, limit=4)
        content = "\n\n".join(f"[{snippet.source}]\n{snippet.content}" for snippet in snippets)
        if not content:
            content = "No matching knowledge was found."
        return ToolResult(name="knowledge_search", content=content)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_arithmetic(expression: str) -> int | float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolValidationError("Invalid arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > 32:
        raise ToolValidationError("Arithmetic expression is too complex")

    def evaluate(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 12:
            raise ToolValidationError("Arithmetic expression is too complex")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise ToolValidationError("Invalid arithmetic value")
            return _bounded_number(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = _UNARY_OPERATORS[type(node.op)](evaluate(node.operand, depth + 1))
            return _bounded_number(value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ToolValidationError("Exponent is too large")
            try:
                value = _BINARY_OPERATORS[type(node.op)](left, right)
            except (ArithmeticError, OverflowError):
                raise ToolValidationError("Arithmetic operation failed") from None
            return _bounded_number(value)
        raise ToolValidationError("Expression contains a disallowed operation")

    return evaluate(tree)


def _bounded_number(value: int | float | complex) -> int | float:
    if isinstance(value, complex) or abs(value) > _MAX_ABSOLUTE_VALUE:
        raise ToolValidationError("Arithmetic result is out of range")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolValidationError("Arithmetic result is out of range")
    return value
