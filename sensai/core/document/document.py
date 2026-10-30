import re
from dataclasses import dataclass, field

DEFAULT_TEMPLATE = """\
# Investigation

## Summary

## People and companies

## Timeline

## Open questions
"""
OPERATIONS = ("append", "replace", "rewrite", "add_section")
# A model that sends more edits than this in one turn is rewriting the document instead of updating it.
MAX_EDITS = 8
# Versions kept for /document undo, one per turn that changed the document.
MAX_HISTORY = 20


class DocumentError(Exception):
    pass


def _key(name: str) -> str:
    return " ".join(name.strip().strip("#").split()).casefold()


@dataclass
class Section:
    name: str
    body: str = ""


@dataclass
class Edit:
    """One change to one section, as the model sends it."""

    op: str
    section: str
    text: str = ""
    # For replace: the passage of the section to change, copied exactly.
    find: str = ""

    def describe(self) -> str:
        return f"{self.op} in '{self.section}'" if self.section else self.op


@dataclass
class Rejected:
    edit: Edit
    reason: str


@dataclass
class EditReport:
    applied: list[Edit] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.applied or self.rejected)


class Document:
    """
    A Markdown deliverable made of `## ` sections, kept apart from the chat history. It only changes through edits
    that each touch one section, so that a turn can never rewrite or corrupt the whole document.
    """

    def __init__(self, markdown: str = DEFAULT_TEMPLATE):
        self.title, self.preamble, self.sections = self._parse(markdown)
        self.history: list[str] = []

    @staticmethod
    def _parse(markdown: str) -> tuple[str, str, list[Section]]:
        title, preamble, sections = "", [], []
        for line in markdown.replace("\r\n", "\n").split("\n"):
            if line.startswith("## "):
                sections.append(Section(line[3:].strip(), ""))
            elif line.startswith("# ") and not title and not sections:
                title = line[2:].strip()
            elif sections:
                sections[-1].body += line + "\n"
            else:
                preamble.append(line)
        for section in sections:
            section.body = section.body.strip("\n")
        return title, "\n".join(preamble).strip("\n"), sections

    def markdown(self) -> str:
        parts = [f"# {self.title}"] if self.title else []
        if self.preamble:
            parts.append(self.preamble)
        for section in self.sections:
            parts.append(f"## {section.name}\n{section.body}" if section.body else f"## {section.name}")
        return "\n\n".join(parts) + "\n"

    @property
    def is_empty(self) -> bool:
        return not self.preamble and not any(section.body for section in self.sections)

    def section(self, name: str) -> Section | None:
        key = _key(name)
        return next((section for section in self.sections if _key(section.name) == key), None)

    def names(self) -> list[str]:
        return [section.name for section in self.sections]

    def apply(self, edits: list[Edit]) -> EditReport:
        """Applies each valid edit and rejects the others with the reason. All of them are one /document undo step."""
        before, report = self.markdown(), EditReport()
        for number, edit in enumerate(edits):
            if number >= MAX_EDITS:
                report.rejected.append(Rejected(edit, f"more than {MAX_EDITS} edits in one turn"))
                continue
            try:
                self._apply(edit)
            except DocumentError as e:
                report.rejected.append(Rejected(edit, str(e)))
            else:
                report.applied.append(edit)
        if report.applied:
            self.history = (self.history + [before])[-MAX_HISTORY:]
        return report

    def _apply(self, edit: Edit) -> None:
        if edit.op not in OPERATIONS:
            raise DocumentError(f"unknown operation '{edit.op}', expected one of {', '.join(OPERATIONS)}")
        name, text = " ".join(edit.section.split()), edit.text.strip("\n")
        if not name:
            raise DocumentError("no section given")
        section = self.section(name)
        if edit.op == "add_section":
            if section is not None:
                raise DocumentError(f"the section '{section.name}' already exists")
            if "\n" in name or name.startswith("#"):
                raise DocumentError("a section name is one line, without #")
            self.sections.append(Section(name, _no_headings(text)))
            return
        if section is None:
            raise DocumentError(f"no section '{name}'; the sections are: {', '.join(self.names())}")
        if edit.op != "replace" and not text.strip():
            raise DocumentError("the text is empty")
        text = _no_headings(text)
        if edit.op == "append":
            if _repeats(text, section.body):
                raise DocumentError("the section already says it")
            section.body = f"{section.body}\n{text}" if section.body else text
        elif edit.op == "rewrite":
            section.body = text
        elif not section.body:
            # Nothing to correct in an empty section: small models still "replace" in it to fill it.
            section.body = text
        else:
            found = section.body.count(edit.find) if edit.find else 0
            if not found:
                raise DocumentError("the text to find is not in the section")
            if found > 1:
                raise DocumentError(f"the text to find is {found} times in the section: quote a longer passage")
            section.body = section.body.replace(edit.find, text).strip("\n")

    def undo(self) -> bool:
        if not self.history:
            return False
        previous = self.history.pop()
        self.title, self.preamble, self.sections = self._parse(previous)
        return True


# A parenthesis that cites where a fact comes from, "(email of 4 April)" or "(report.pdf, p. 3)". Any other
# parenthesis is part of the fact: "Marc Vidal (CEO)" and "Marc Vidal (CFO)" are two different facts.
CITATION = re.compile(
    r"\((?:e-?mail|mail|source|page|p\.|doc|document|file|report|interview|article|according|see|cf)\b[^)]*\)"
    r"|\([^)]*\.\w{2,4}\b[^)]*\)",
    re.IGNORECASE,
)


def _fact(line: str) -> str:
    # The same fact cited another way, "[mail of 4 April]" or "(email of 4 April)", is still the same fact.
    line = CITATION.sub(" ", re.sub(r"\[[^\]]*\]", " ", line))
    return " ".join(re.findall(r"\w+", line.casefold()))


def _repeats(text: str, body: str) -> bool:
    """True when every line of the text is already in the body: appending it would only add a duplicate."""
    known = {_fact(line) for line in body.split("\n")}
    facts = [_fact(line) for line in text.split("\n") if _fact(line)]
    return bool(facts) and all(fact in known for fact in facts)


def _no_headings(text: str) -> str:
    # A `## ` line in a section body would split it in two the next time the document is read.
    return re.sub(r"^(#{1,2}) ", r"### ", text, flags=re.M)
