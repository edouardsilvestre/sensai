from dataclasses import dataclass

from sensai.core.cache import CachedAnswer
from sensai.core.document.document import EditReport
from sensai.core.evaluation.judge import Judgement
from sensai.core.examples import Example
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
class RerankEvent:
    """The candidates read by the reranker: `kept` were sent to the model, `dropped` were under the threshold."""

    candidates: int
    kept: int
    dropped: list[SearchResult]


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
class ThoughtEvent:
    number: int
    thought: str


@dataclass
class ReasoningStoppedEvent:
    reason: str


@dataclass
class ExamplesEvent:
    examples: list[Example]


@dataclass
class ExamplesFailedEvent:
    message: str


@dataclass
class LogFailedEvent:
    message: str


@dataclass
class CheckStartedEvent:
    sentences: int
    model: str


@dataclass
class CheckEvent:
    judgement: Judgement


@dataclass
class CheckFailedEvent:
    message: str


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


@dataclass
class CacheHitEvent:
    cached: CachedAnswer


@dataclass
class CacheFailedEvent:
    message: str


Event = (
    SourcesEvent
    | RerankEvent
    | TokenEvent
    | ToolCallEvent
    | ApprovalEvent
    | ToolResultEvent
    | GuardrailEvent
    | ThoughtEvent
    | ReasoningStoppedEvent
    | ExamplesEvent
    | ExamplesFailedEvent
    | DoneEvent
    | LogFailedEvent
    | CheckStartedEvent
    | CheckEvent
    | CheckFailedEvent
    | DocumentEvent
    | DocumentFailedEvent
    | AgentStartedEvent
    | ReviewDoneEvent
    | CompactionStartedEvent
    | CompactionEvent
    | CompactionFailedEvent
    | CacheHitEvent
    | CacheFailedEvent
)
