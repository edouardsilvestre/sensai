import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sensai.core.llm.ollama import ToolCall


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    """`content` is sent to the model, `summary` is only shown to the user."""

    content: str
    summary: str = ""
    error: bool = False


@dataclass
class SchemaTool:
    """
    A tool that comes with its JSON schema, like the tools of an MCP server.
    `run` gets the arguments as they are: whoever runs the tool checks them against the schema.
    """

    name: str
    description: str
    parameters: dict
    run: Callable[[dict], str | ToolResult]
    approval_reason: str | None = None

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


# A tool is a plain function: Ollama builds its schema from the type hints and the docstring (Args, Returns).
# It returns the text for the model, or a ToolResult to also give the user a summary.
# A SchemaTool brings its schema instead.
Tool = Callable[..., str | ToolResult] | SchemaTool


def _name(tool: Tool) -> str:
    return tool.name if isinstance(tool, SchemaTool) else tool.__name__


def requires_approval(reason: str) -> Callable[[Tool], Tool]:
    def mark(tool: Tool) -> Tool:
        tool.approval_reason = reason
        return tool

    return mark


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = (), warnings: Iterable[str] = ()):
        self._tools = {_name(tool): tool for tool in tools}
        # Why tools were left out while the registry was built, like those of an MCP server that is down.
        self.warnings = list(warnings)

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Callable | dict]:
        """The tools as Ollama takes them: the functions, and the schema of each SchemaTool."""
        return [tool.schema() if isinstance(tool, SchemaTool) else tool for tool in self._tools.values()]

    def descriptions(self) -> dict[str, str]:
        """The first paragraph of the description of each tool, for the user."""
        descriptions = {}
        for name, tool in self._tools.items():
            text = tool.description if isinstance(tool, SchemaTool) else inspect.getdoc(tool)
            descriptions[name] = " ".join((text or "").strip().split("\n\n")[0].split())
        return descriptions

    def approval_reason(self, name: str) -> str | None:
        return getattr(self._tools.get(name), "approval_reason", None)

    def prepare(self) -> list[str]:
        """
        Calls `prepare()` on the object of each tool that has one, like the CodeRunner that pulls its Docker image.
        A tool whose preparation fails is removed. Returns why each removed tool is disabled.
        """
        errors = []
        for name, tool in list(self._tools.items()):
            prepare = getattr(getattr(tool, "__self__", None), "prepare", None)
            if prepare is None:
                continue
            try:
                prepare()
            except ToolError as e:
                del self._tools[name]
                errors.append(f"{name} disabled: {e}")
        return errors

    def execute(self, call: ToolCall) -> ToolResult:
        # Failures go back to the model as the tool's answer, so that it can fix its call or answer without it.
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(f"Unknown tool '{call.name}'. Available tools: {', '.join(self._tools) or 'none'}")
        if not isinstance(tool, SchemaTool):
            try:
                inspect.signature(tool).bind(**call.arguments)
            except TypeError as e:
                return _error(f"{call.name}: {e}")
        try:
            output = tool.run(call.arguments) if isinstance(tool, SchemaTool) else tool(**call.arguments)
        except ToolError as e:
            return _error(str(e))
        return output if isinstance(output, ToolResult) else ToolResult(str(output))


def _error(message: str) -> ToolResult:
    return ToolResult(f"Error: {message}", error=True)
