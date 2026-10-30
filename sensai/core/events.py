from dataclasses import dataclass

from sensai.core.guardrails.rules import Hit
from sensai.core.approval import Approval
from sensai.core.llm.ollama import ToolCall
from sensai.core.persona import Persona
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
class AgentStartedEvent:
    """In a review, the agent whose events follow."""

    persona: Persona


@dataclass
class ReviewDoneEvent:
    """The text kept in the history: the editor's when the fact-checker approved, else the last answer."""

    approved: bool
    content: str
    rounds: int


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
    | AgentStartedEvent
    | ReviewDoneEvent
    | CompactionStartedEvent
    | CompactionEvent
    | CompactionFailedEvent
)
