import pytest

from chatbot.processes.project_brief import build_project_brief_plugin
from chatbot.processes.registry import ProcessRegistry


def test_process_registry_rejects_duplicate_names() -> None:
    plugin = build_project_brief_plugin()

    with pytest.raises(ValueError, match="Duplicate process plug-in"):
        ProcessRegistry([plugin, plugin])
