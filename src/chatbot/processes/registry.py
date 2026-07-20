"""Allow-listed process registry constructed at application startup."""

from collections.abc import Iterable

from chatbot.processes.project_brief import build_project_brief_plugin
from chatbot.processes.schemas import ProcessPlugin
from chatbot.processes.troubleshoot import build_troubleshoot_plugin
from chatbot.retrieval.models import KnowledgeBase


class ProcessRegistry:
    def __init__(self, plugins: Iterable[ProcessPlugin]) -> None:
        registered: dict[str, ProcessPlugin] = {}
        for plugin in plugins:
            initial_record = plugin.start()
            if initial_record.name != plugin.name:
                raise ValueError("Process plug-in name does not match its initial record")
            if plugin.name in registered:
                raise ValueError(f"Duplicate process plug-in: {plugin.name}")
            if not plugin.description.strip() or len(plugin.description) > 300:
                raise ValueError("Process descriptions must contain 1 to 300 characters")
            registered[plugin.name] = plugin
        if not registered:
            raise ValueError("At least one process plug-in must be registered")
        if len(registered) > 20:
            raise ValueError("At most 20 process plug-ins may be registered")
        self._plugins = registered

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._plugins)

    def get(self, name: str) -> ProcessPlugin | None:
        return self._plugins.get(name)

    def descriptions(self) -> tuple[tuple[str, str], ...]:
        return tuple((name, plugin.description) for name, plugin in self._plugins.items())


def build_process_registry(knowledge_base: KnowledgeBase) -> ProcessRegistry:
    """Build the reviewed process allow-list once per application lifespan."""

    return ProcessRegistry(
        [
            build_project_brief_plugin(),
            build_troubleshoot_plugin(knowledge_base),
        ]
    )
