import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from sensai.core.document.document import Document
from sensai.core.export.page import page
from sensai.core.guardrails.rules import Guardrails, Hit
from sensai.core.memory.conversation import ConversationMemory, Turn
from sensai.core.memory.sessions import conversation_title, short_title
from sensai.core.rag.prompt import cited_numbers

YOU, SENSAI = "You", "Sensai"
SUMMARY = "Summary of the older messages"
HIDDEN_SOURCE = "[source hidden by a rule]"
FORMATS = {
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".html": "text/html; charset=utf-8",
}
# Up to the last separator: a folder can name the case or the source, and the page only has to say which file.
_FOLDERS = re.compile(r".*[\\/]")
_WORD_SEPARATORS = re.compile(r"[_.-]+")


class ExportError(Exception):
    pass


@dataclass
class Part:
    """A piece of a report: who wrote it, its Markdown, and the sources it cites with their numbers."""

    author: str
    text: str
    sources: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class Report:
    """A conversation or a document, in no format yet. `data` is its JSON, `markdown` its Markdown if it has one."""

    title: str
    details: str
    parts: list[Part]
    data: dict = field(default_factory=dict)
    markdown: str = ""


def _cited(turn: Turn) -> list[tuple[int, str]]:
    numbers = [number for number in sorted(cited_numbers(turn.assistant)) if 1 <= number <= len(turn.citations)]
    return [(number, turn.citations[number - 1]) for number in numbers]


def conversation_report(memory: ConversationMemory, model: str, data: dict) -> Report:
    """The active branch, with what is left of the condensed turns. `data` is the saved session: all the branches."""
    parts = [Part(SUMMARY, memory.summary)] if memory.summary else []
    for turn in memory.turns:
        parts += [Part(YOU, turn.user), Part(SENSAI, turn.assistant, _cited(turn))]
    branches = memory.branches()
    branch = [f"branch {memory.branch.number} of {len(branches)}"] if len(branches) > 1 else []
    details = ", ".join([f"Model {model}", *branch, f"{datetime.now():%Y-%m-%d %H:%M}"])
    return Report(conversation_title(memory.turns), details, parts, data)


def document_report(document: Document) -> Report:
    markdown = document.markdown()
    data = {
        "title": document.title, "preamble": document.preamble,
        "sections": [{"name": section.name, "body": section.body} for section in document.sections],
    }
    body = markdown.removeprefix(f"# {document.title}\n\n")
    title = document.title or "Working document"
    return Report(title, f"Working document, {datetime.now():%Y-%m-%d %H:%M}", [Part("", body)], data, markdown)


def to_markdown(report: Report) -> str:
    if report.markdown:
        return report.markdown
    lines = [f"# {report.title}", "", f"*{report.details}*"]
    for part in report.parts:
        lines += ["", f"## {part.author}", "", part.text.strip()]
        if part.sources:
            lines += ["", *(f"- [{number}] {citation}" for number, citation in part.sources)]
    return "\n".join(lines) + "\n"


def _parts(report: Report) -> list[tuple[str, str, list[tuple[int, str]]]]:
    return [(part.author, part.text, part.sources) for part in report.parts]


def to_html(report: Report) -> str:
    return page(report.title, report.details, _parts(report), YOU)


def render(report: Report, suffix: str) -> bytes:
    if suffix == ".json":
        return json.dumps(report.data, ensure_ascii=False, indent=2).encode("utf-8")
    if suffix == ".html":
        return to_html(report).encode("utf-8")
    if suffix == ".pdf":
        # Here: without it, everything but the PDF still works.
        try:
            from sensai.core.export.pdf import to_pdf
        except ImportError as e:
            raise ExportError("The PDF export needs reportlab: pip install -r requirements.txt") from e
        return to_pdf(report.title, report.details, _parts(report))
    return to_markdown(report).encode("utf-8")


@dataclass
class Publication:
    """A report as a public page would show it, and what the rules did to it. `blocked` stops the publication."""

    report: Report
    hits: list[Hit] = field(default_factory=list)
    blocked: Hit | None = None

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(citation for part in self.report.parts for _, citation in part.sources))


def prepare_publication(report: Report, guardrails: Guardrails) -> Publication:
    """Puts a report through the rules, and keeps of its sources only the file: only `parts` and `title` are published."""
    hits: dict[str, Hit] = {}
    blocked: list[Hit] = []

    def clean(text: str) -> str:
        checked = guardrails.check_publication(text)
        blocked.extend([checked.blocked] if checked.blocked else [])
        for hit in checked.hits:
            kept = hits.setdefault(hit.rule, Hit(hit.stage, hit.rule, hit.action, 0, "", hit.message))
            kept.count += hit.count
            kept.found = ", ".join(dict.fromkeys(filter(None, [*kept.found.split(", "), *hit.found.split(", ")])))
        return checked.text

    def source(citation: str) -> str:
        name = _FOLDERS.sub("", citation)
        # A file is called claire_benoit.eml, and a rule looks for the words "Claire Benoit".
        words = _WORD_SEPARATORS.sub(" ", name)
        return name if clean(words) == words else HIDDEN_SOURCE

    parts = [
        Part(part.author, clean(part.text), [(number, source(citation)) for number, citation in part.sources])
        for part in report.parts
    ]
    # From the redacted message: a title cut in the middle of a protected name would leave a piece no rule can find.
    first = next((part.text for part in parts if part.author == YOU), "")
    title = short_title(first) if first else clean(report.title)
    return Publication(Report(title, report.details, parts), list(hits.values()), next(iter(blocked), None))
