import difflib
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sensai.core.config import Config
from sensai.core.llm.ollama import OllamaClient, Usage

COMPARE_OPTIONS = {"temperature": 0, "seed": 42}


class PromptError(Exception):
    pass


@dataclass
class PromptVersion:
    number: int
    prompt: str
    created_at: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.prompt.encode()).hexdigest()[:8]


@dataclass
class PromptHistory:
    versions: list[PromptVersion] = field(default_factory=list)
    active: int = 0
    comparisons: list[dict] = field(default_factory=list)


def _parse(data: object) -> PromptHistory:
    if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
        raise ValueError('expected {"versions": [...]}')
    versions = []
    for number, item in enumerate(data["versions"], start=1):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"version {number} has no prompt")
        versions.append(PromptVersion(number, item["prompt"], str(item.get("created_at", ""))))
    active = data.get("active", len(versions))
    if not isinstance(active, int) or not (versions and 1 <= active <= len(versions)):
        raise ValueError(f"the active version {active} does not exist")
    comparisons = data.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError("the comparisons are not a list")
    return PromptHistory(versions, active, comparisons)


class PromptStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self._history: PromptHistory | None = None

    @property
    def path(self) -> Path:
        return self.directory / "system.json"

    def history(self) -> PromptHistory:
        if self._history is None:
            if not self.path.exists():
                return PromptHistory()
            try:
                self._history = _parse(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError) as e:
                raise PromptError(f"Cannot read the prompt versions {self.path}: {e}") from None
            except ValueError as e:
                raise PromptError(f"Invalid prompt versions file {self.path}: {e}") from None
        return self._history

    def snapshot(self, prompt: str) -> PromptVersion | None:
        history = self.history()
        if history.versions and history.versions[-1].prompt == prompt:
            return None
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        version = PromptVersion(len(history.versions) + 1, prompt, created)
        self._save(PromptHistory(history.versions + [version], version.number, history.comparisons))
        return version

    def version(self, number: int) -> PromptVersion:
        versions = self.history().versions
        if not 1 <= number <= len(versions):
            raise PromptError(f"No version {number} of system prompt: it has {len(versions)}")
        return versions[number - 1]

    def active(self) -> PromptVersion | None:
        history = self.history()
        return history.versions[history.active - 1] if history.versions else None

    def activate(self, number: int) -> PromptVersion:
        version = self.version(number)
        history = self.history()
        self._save(PromptHistory(history.versions, number, history.comparisons))
        return version

    def record(self, comparison: dict) -> None:
        history = self.history()
        self._save(PromptHistory(history.versions, history.active, history.comparisons + [comparison]))

    def _save(self, history: PromptHistory) -> None:
        path = self.path
        temporary = path.with_name(path.name + ".tmp")
        data = {
            "active": history.active,
            "versions": [{"prompt": version.prompt, "created_at": version.created_at} for version in history.versions],
            "comparisons": history.comparisons,
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except OSError as e:
            raise PromptError(f"Cannot save the prompt versions {path}: {e}") from None
        self._history = history


def diff(old: PromptVersion, new: PromptVersion) -> str:
    lines = difflib.unified_diff(
        old.prompt.splitlines(), new.prompt.splitlines(), f"version {old.number}", f"version {new.number}", lineterm=""
    )
    return "\n".join(lines)


@dataclass
class Case:
    question: str
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case: Case
    answer: str
    seconds: float
    tokens: int
    failed: list[str]


@dataclass
class VersionScore:
    version: int
    results: list[CaseResult]

    @property
    def passed(self) -> int:
        return sum(not result.failed for result in self.results)

    @property
    def seconds(self) -> float:
        return sum(result.seconds for result in self.results) / len(self.results)

    @property
    def tokens(self) -> float:
        return sum(result.tokens for result in self.results) / len(self.results)

    def summary(self) -> dict:
        return {
            "version": self.version, "passed": self.passed, "cases": len(self.results),
            "seconds": round(self.seconds, 2), "tokens": round(self.tokens),
        }


def _texts(value: object, name: str, number: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(text, str) and text.strip() for text in value):
        raise ValueError(f"'{name}' of case {number} must be a list of texts")
    return value


def load_cases(path: Path) -> list[Case]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PromptError(f"No test cases at {path}: see prompt_cases.example.json") from None
    except (OSError, UnicodeDecodeError) as e:
        raise PromptError(f"Cannot read the test cases {path}: {e}") from None
    except ValueError as e:
        raise PromptError(f"Invalid test cases file {path}: {e}") from None
    try:
        if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
            raise ValueError('expected {"cases": [...]} with at least one case')
        cases = []
        for number, item in enumerate(data["cases"], start=1):
            if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not item["question"].strip():
                raise ValueError(f"case {number} needs a question")
            cases.append(Case(
                item["question"],
                _texts(item.get("must_contain", []), "must_contain", number),
                _texts(item.get("must_not_contain", []), "must_not_contain", number),
            ))
    except ValueError as e:
        raise PromptError(f"Invalid test cases file {path}: {e}") from None
    return cases


def check(case: Case, answer: str) -> list[str]:
    text = answer.lower()
    missing = [f'missing "{expected}"' for expected in case.must_contain if expected.lower() not in text]
    return missing + [f'contains "{banned}"' for banned in case.must_not_contain if banned.lower() in text]


def score(
    client: OllamaClient, config: Config, version: PromptVersion, cases: list[Case],
    clock: Callable[[], float] = time.perf_counter,
) -> VersionScore:
    options = {"num_ctx": config.num_ctx, **COMPARE_OPTIONS}
    results = []
    for case in cases:
        usage = Usage()
        messages = [{"role": "system", "content": version.prompt}, {"role": "user", "content": case.question}]
        start = clock()
        answer = client.chat(config.model, messages, options, think=config.think, usage=usage)
        results.append(CaseResult(case, answer, clock() - start, usage.total_tokens, check(case, answer)))
    return VersionScore(version.number, results)


def compare(client: OllamaClient, config: Config, versions: list[PromptVersion]) -> list[VersionScore]:
    cases = load_cases(config.prompt_cases_path)
    return [score(client, config, version, cases) for version in versions]


def comparison(model: str, scores: list[VersionScore]) -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "scores": [score.summary() for score in scores],
    }
