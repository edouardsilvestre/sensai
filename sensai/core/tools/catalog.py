from sensai.core.config import Config
from sensai.core.llm.ollama import OllamaClient
from sensai.core.memory.store import MemoryStore
from sensai.core.tools.code_runner import CodeRunner
from sensai.core.tools.files import FileTools
from sensai.core.tools.memory import MemoryTools
from sensai.core.tools.permissions import Permissions
from sensai.core.tools.registry import ToolRegistry
from sensai.core.tools.web_search import OllamaSearchBackend, WebSearch


def build_registry(config: Config, client: OllamaClient) -> ToolRegistry:
    """The tools enabled in the configuration. A new tool is a documented function in this package plus one line here."""
    tools = []
    if not config.features["tools"]:
        return ToolRegistry()
    if config.features["web_search"]:
        tools.append(WebSearch(OllamaSearchBackend(client), config.web_search_results).web_search)
    if config.features["code_exec"]:
        tools.append(CodeRunner(config.sandbox_image, config.code_timeout).run_python)
    if config.features["memory"]:
        tools += MemoryTools(MemoryStore(config.memory_path)).tools()
    if config.features["files"]:
        tools += FileTools(Permissions(config.files_read, config.files_write, config.files_deny)).tools()
    return ToolRegistry(tools)
