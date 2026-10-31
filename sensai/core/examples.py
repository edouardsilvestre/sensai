import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers


_INSTRUCTIONS = """\
Below are answers the user kept as good examples, to questions close to the next message. Answer it the same way:
- cite the excerpts that support each claim, like the examples do;
- say plainly what the documents do not say, instead of guessing;
- keep the established facts apart from the hypotheses.
The facts of the examples belong to other questions: never use them to answer the next message."""


class ExampleError(Exception):
    pass


@dataclass
class Excerpt:
    number: int
    source: str
    location: str
    text: str


@dataclass
class Example:
    question: str
    answer: str
    excerpts: list[Excerpt] = field(default_factory=list)
    created_at: str = ""


def make_example(question: str, answer: str, sources: list[SearchResult]) -> Example:
    question, answer = question.strip(), answer.strip()
    if not question or not answer:
        raise ExampleError("An example needs a question and its answer")
    excerpts = [
        Excerpt(number, sources[number - 1].source, sources[number - 1].location, sources[number - 1].text)
        for number in sorted(cited_numbers(answer))
        if 1 <= number <= len(sources)
    ]
    return Example(question, answer, excerpts, datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _escape(text: str) -> str:
    return text.replace("</example", "<\\/example").replace("</excerpt", "<\\/excerpt")


def _attribute(value: str) -> str:
    return value.replace('"', "'")


def _render(example: Example) -> str:
    excerpts = []
    for excerpt in example.excerpts:
        attributes = f'id="{excerpt.number}" source="{_attribute(excerpt.source)}"'
        if excerpt.location:
            attributes += f' location="{_attribute(excerpt.location)}"'
        excerpts.append(f"<excerpt {attributes}>\n{_escape(excerpt.text)}\n</excerpt>")
    block = "<excerpts>\n" + "\n".join(excerpts) + "\n</excerpts>\n" if excerpts else ""
    return f"<example>\n{block}Message: {_escape(example.question)}\nAnswer: {_escape(example.answer)}\n</example>"


def examples_message(examples: list[Example]) -> str:
    return _INSTRUCTIONS + "\n\n" + "\n".join(_render(example) for example in examples)


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def _excerpt(data: dict) -> Excerpt:
    return Excerpt(int(data["number"]), str(data["source"]), str(data.get("location", "")), str(data["text"]))


def _parse(data: object) -> list[Example]:
    if not isinstance(data, dict) or not isinstance(data.get("examples"), list):
        raise ValueError('expected {"examples": [...]}')
    examples = []
    for number, item in enumerate(data["examples"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"example {number} is not an object")
        question, answer, excerpts = item.get("question"), item.get("answer"), item.get("excerpts", [])
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"example {number} needs a question and an answer")
        if not isinstance(excerpts, list):
            raise ValueError(f"the excerpts of example {number} are not a list")
        try:
            parsed = [_excerpt(excerpt) for excerpt in excerpts]
        except (TypeError, KeyError, ValueError):
            raise ValueError(f"example {number} has an excerpt without a number, a source or a text") from None
        examples.append(Example(question, answer, parsed, str(item.get("created_at", ""))))
    return examples


class ExampleLibrary:
    def __init__(self, path: Path, embed: Callable[[str], list[float]], min_similarity: float = 0.5):
        self.path = path
        self.embed = embed
        self.min_similarity = min_similarity
        self.examples = self._read()
        self._vectors: list[list[float]] = []

    def _read(self) -> list[Example]:
        if not self.path.exists():
            return []
        try:
            return _parse(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as e:
            raise ExampleError(f"Cannot read the examples {self.path}: {e}") from None
        except ValueError as e:
            raise ExampleError(f"Invalid examples file {self.path}: {e}") from None

    def closest(self, question: str, k: int) -> list[Example]:
        if not self.examples or k <= 0:
            return []
        while len(self._vectors) < len(self.examples):
            self._vectors.append(self.embed(self.examples[len(self._vectors)].question))
        vector = self.embed(question)
        scored = sorted(
            ((_cosine(vector, other), number) for number, other in enumerate(self._vectors)), reverse=True
        )
        return [self.examples[number] for similarity, number in scored[:k] if similarity >= self.min_similarity]

    def add(self, example: Example) -> None:
        for number, kept in enumerate(self.examples, start=1):
            if (kept.question, kept.answer) == (example.question, example.answer):
                raise ExampleError(f"This answer is already kept as example {number}")
        self.examples.append(example)
        self._save()

    def remove(self, number: int) -> Example:
        if not 1 <= number <= len(self.examples):
            raise ExampleError(f"No example {number}: the library has {len(self.examples)}")
        example = self.examples.pop(number - 1)
        if len(self._vectors) >= number:
            self._vectors.pop(number - 1)
        self._save()
        return example

    def _save(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        data = {"examples": [asdict(example) for example in self.examples]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError as e:
            raise ExampleError(f"Cannot save the examples {self.path}: {e}") from None
