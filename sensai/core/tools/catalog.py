from sensai.core.config import Config
from sensai.core.llm.ollama import OllamaClient
from sensai.core.tools.registry import ToolRegistry
from sensai.core.tools.web_search import OllamaSearchBackend, WebSearch


def build_registry(config: Config, client: OllamaClient) -> ToolRegistry:
    """The tools enabled in the configuration. A new tool is a documented function in this package plus one line here."""
    tools = []
    if not config.features["tools"]:
        return ToolRegistry()
    if config.features["web_search"]:
        tools.append(WebSearch(OllamaSearchBackend(client), config.web_search_results).web_search)
    return ToolRegistry(tools)
