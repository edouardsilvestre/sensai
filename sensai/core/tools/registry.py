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


# A tool is a plain function: Ollama builds its schema from the type hints and the docstring (Args, Returns).
# It returns the text for the model, or a ToolResult to also give the user a summary.
Tool = Callable[..., str | ToolResult]


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools = {tool.__name__: tool for tool in tools}

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        # Failures go back to the model as the tool's answer, so that it can fix its call or answer without it.
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(f"Unknown tool '{call.name}'. Available tools: {', '.join(self._tools) or 'none'}")
        try:
            inspect.signature(tool).bind(**call.arguments)
        except TypeError as e:
            return _error(f"{call.name}: {e}")
        try:
            output = tool(**call.arguments)
        except ToolError as e:
            return _error(str(e))
        return output if isinstance(output, ToolResult) else ToolResult(str(output))


def _error(message: str) -> ToolResult:
    return ToolResult(f"Error: {message}", error=True)
