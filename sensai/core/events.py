from dataclasses import dataclass

from sensai.core.rag.models import SearchResult


@dataclass
class SourcesEvent:
    sources: list[SearchResult]


@dataclass
class TokenEvent:
    text: str


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
    | DoneEvent
    | CompactionStartedEvent
    | CompactionEvent
    | CompactionFailedEvent
)
