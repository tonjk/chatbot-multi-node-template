import pytest

from chatbot.processes.number_counter import build_number_counter_plugin
from chatbot.processes.registry import ProcessRegistry


class FakeProcessModel:
    def extract_numbers(self, message: str) -> dict[str, list[int]]:
        return {"numbers": []}


def test_process_registry_rejects_duplicate_names() -> None:
    plugin = build_number_counter_plugin(FakeProcessModel())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Duplicate process plug-in"):
        ProcessRegistry([plugin, plugin])
