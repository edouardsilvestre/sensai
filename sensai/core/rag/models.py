import re
import shlex
from dataclasses import astuple, dataclass, field, fields
from datetime import date


@dataclass(frozen=True)
class Locator:
    kind: str
    number: int
    scope: str = ""


@dataclass
class Section:
    text: str
    locator: Locator | None = None
    separator: str = "\n\n"


@dataclass
class Document:
    kind: str
    sections: list[Section]
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    location: str = ""
    overlap: int = 0


@dataclass
class SearchResult:
    chunk_id: int
    text: str
    location: str
    source: str
    description: str
    # The cosine similarity of the chunk to the query.
    score: float
    # From 0 to 1, how well the chunk answers the query according to the reranker; None when it was not reranked.
    relevance: float | None = None

    @property
    def citation(self) -> str:
        return f"{self.source}, {self.location}" if self.location else self.source


@dataclass
class SearchReport:
    """What the reranker did with the candidates of a search, filled by the search when it is given one."""

    # The chunks the reranker read; 0 when the search was not reranked.
    candidates: int = 0
    # The chunks under the relevance threshold, the most relevant first.
    dropped: list[SearchResult] = field(default_factory=list)
    # Why the search was not reranked although the feature is on.
    rerank_error: str = ""


_DESCRIBED_FIELDS = {"from": "From", "to": "To", "cc": "Cc", "date": "Date", "author": "Author", "attachments": "Attachments"}


def describe(title: str, metadata: dict[str, str]) -> str:
    lines = [f"Title: {title}"] if title else []
    lines += [f"{label}: {metadata[key]}" for key, label in _DESCRIBED_FIELDS.items() if metadata.get(key)]
    return "\n".join(lines)


KINDS = ("email", "html", "pdf", "spreadsheet", "text")
# The key of each field in the filter syntax, like "from:dupont".
_FILTER_KEYS = {"after": "after", "before": "before", "author": "from", "recipient": "to", "kind": "kind", "source": "source"}
_PERIOD = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")


@dataclass(frozen=True)
class SearchFilter:
    """Limits a search to the documents that match every field set. after is included, before is not."""

    after: str = ""
    before: str = ""
    author: str = ""
    recipient: str = ""
    kind: str = ""
    # The start of the document name, like "leak/mails/".
    source: str = ""

    def __post_init__(self):
        for name in ("after", "before"):
            value = getattr(self, name)
            if value and not _is_period(value):
                raise ValueError(f"{name} must be a year, a month or a day like 2023, 2023-03 or 2023-03-14, got '{value}'")
        if self.after and self.before and self.after >= self.before:
            raise ValueError(f"after ({self.after}) must be earlier than before ({self.before})")
        if self.kind and self.kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}, got '{self.kind}'")

    def is_empty(self) -> bool:
        return not any(astuple(self))

    def describe(self) -> str:
        parts = []
        for item in fields(self):
            value = getattr(self, item.name)
            if value:
                quoted = f'"{value}"' if re.search(r"[\s'\"]", value) else value
                parts.append(f"{_FILTER_KEYS[item.name]}:{quoted}")
        return " ".join(parts)


def _is_period(value: str) -> bool:
    if not _PERIOD.fullmatch(value):
        return False
    year, month, day = (value.split("-") + ["01", "01"])[:3]
    try:
        date(int(year), int(month), int(day))
    except ValueError:
        return False
    return True


def parse_filter(text: str) -> SearchFilter:
    """Reads "after:2023-03 from:\"Jean Dupont\" kind:email"."""
    keys = {key: name for name, key in _FILTER_KEYS.items()}
    values: dict[str, str] = {}
    try:
        words = shlex.split(text)
    except ValueError as e:
        raise ValueError(f"Cannot read the filter: {e}") from None
    for word in words:
        key, _, value = word.partition(":")
        if not value.strip():
            raise ValueError(f"Write each filter as KEY:VALUE, like after:2023-03, got '{word}'")
        if key.lower() not in keys:
            raise ValueError(f"Unknown filter '{key}'. Use {', '.join(keys)}")
        name = keys[key.lower()]
        values[name] = value.strip().lower() if name == "kind" else value.strip()
    return SearchFilter(**values)
