from dataclasses import dataclass

from sensai.core.llm.ollama import ToolCall
from sensai.core.rag.models import SearchResult
from sensai.core.tools.registry import ToolResult


@dataclass
class SourcesEvent:
    sources: list[SearchResult]


@dataclass
class TokenEvent:
    text: str


@dataclass
class ToolCallEvent:
    call: ToolCall


@dataclass
class ToolResultEvent:
    call: ToolCall
    result: ToolResult


@dataclass
class DoneEvent:
    content: str


@dataclass
class CompactionStartedEvent:
    tokens: int


@dataclass
class CompactionEvent:
    turns: int
    tokens: int


@dataclass
class CompactionFailedEvent:
    message: str


Event = (
    SourcesEvent
    | TokenEvent
    | ToolCallEvent
    | ToolResultEvent
    | DoneEvent
    | CompactionStartedEvent
    | CompactionEvent
    | CompactionFailedEvent
)
