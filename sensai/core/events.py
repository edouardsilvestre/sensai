from dataclasses import dataclass

from sensai.core.document.document import EditReport
from sensai.core.guardrails.rules import Hit
from sensai.core.approval import Approval
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
class ApprovalEvent:
    call: ToolCall
    approval: Approval


@dataclass
class ToolResultEvent:
    call: ToolCall
    result: ToolResult


@dataclass
class GuardrailEvent:
    """A rule found something: in the message (input), a tool call (tools) or the answer (output)."""

    hit: Hit


@dataclass
class DoneEvent:
    content: str


@dataclass
class DocumentEvent:
    """The edits applied to the working document after the answer, and the ones rejected with their reason."""

    report: EditReport


@dataclass
class DocumentFailedEvent:
    message: str


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
    | ApprovalEvent
    | ToolResultEvent
    | GuardrailEvent
    | DoneEvent
    | DocumentEvent
    | DocumentFailedEvent
    | CompactionStartedEvent
    | CompactionEvent
    | CompactionFailedEvent
)
