import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from sensai.core.agent import Agent
from sensai.core.config import Config
from sensai.core.evaluation.facts import facts, known, plain
from sensai.core.evaluation.judge import Judgement, check, evidence, sentences
from sensai.core.events import DoneEvent, SourcesEvent, ToolResultEvent
from sensai.core.llm.ollama import LLMError

TURNED_OFF = ("fact_check", "document", "cache", "examples")


class EvalError(Exception):
    pass


@dataclass
class Case:
    question: str
    expected: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    question: str
    answer: str = ""
    seconds: float = 0.0
    expected: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    relevance: int | None = None
    coherence: int | None = None
    flagged: list[dict] = field(default_factory=list)
    unchecked: int = 0
    error: str = ""


@dataclass
class Summary:
    cases: int
    errors: int
    expected: float | None
    faithfulness: float | None
    relevance: float | None
    coherence: float | None
    flagged: int
    seconds: float | None


def load_cases(path: Path) -> list[Case]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise EvalError(f"Cannot read the questions {path}: {e}") from None
    except json.JSONDecodeError as e:
        raise EvalError(f"Invalid questions file {path}: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise EvalError(f'Invalid questions file {path}: expected {{"cases": [...]}} with at least one question')
    cases = []
    for number, item in enumerate(data["cases"], start=1):
        question = item.get("question") if isinstance(item, dict) else None
        expected = item.get("expected", []) if isinstance(item, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise EvalError(f"Invalid questions file {path}: question {number} is empty")
        if not isinstance(expected, list) or not all(isinstance(fact, str) and fact.strip() for fact in expected):
            raise EvalError(f"Invalid questions file {path}: the expected facts of question {number} are not texts")
        cases.append(Case(question.strip(), [fact.strip() for fact in expected]))
    return cases


def contains(answer: str, expected: str) -> bool:
    wanted = [fact for fact in facts(expected) if fact.kind != "name"]
    if wanted:
        values, _ = known(answer)
        return all(fact.value in values for fact in wanted)
    return re.search(rf"\b{re.escape(plain(expected))}\b", plain(answer)) is not None


def eval_config(config: Config) -> Config:
    return replace(config, features={**config.features, **{name: False for name in TURNED_OFF}})


def run_case(agent: Agent, config: Config, case: Case) -> CaseResult:
    result = CaseResult(case.question, expected=list(case.expected))
    sources, results = [], []
    start = time.monotonic()
    try:
        for event in agent.run(case.question):
            if isinstance(event, SourcesEvent):
                sources = event.sources
            elif isinstance(event, ToolResultEvent) and not event.result.error:
                if event.call.name != "search_documents":
                    results.append(event.result.content)
            elif isinstance(event, DoneEvent):
                result.answer = event.content.strip()
                result.seconds = round(time.monotonic() - start, 1)
    except LLMError as e:
        result.error = str(e)
        return result
    result.found = [fact for fact in case.expected if contains(result.answer, fact)]
    if not sentences(result.answer):
        return result
    try:
        judgement = check(agent.client, config, case.question, result.answer, evidence(sources, results))
    except LLMError as e:
        result.error = f"The answer could not be judged: {e}"
        return result
    _keep(result, judgement)
    return result


def _keep(result: CaseResult, judgement: Judgement) -> None:
    result.faithfulness, result.relevance, result.coherence = (
        judgement.faithfulness, judgement.relevance, judgement.coherence
    )
    result.flagged = [asdict(claim) for claim in judgement.flagged]
    result.unchecked = len(judgement.unchecked)


def _mean(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def summarize(results: list[CaseResult]) -> Summary:
    expected = sum(len(result.expected) for result in results)
    return Summary(
        cases=len(results),
        errors=sum(bool(result.error) for result in results),
        expected=sum(len(result.found) for result in results) / expected if expected else None,
        faithfulness=_mean([result.faithfulness for result in results]),
        relevance=_mean([result.relevance for result in results]),
        coherence=_mean([result.coherence for result in results]),
        flagged=sum(len(result.flagged) for result in results),
        seconds=_mean([result.seconds for result in results if not result.error]),
    )


def run_eval(
    make_agent: Callable[[], Agent], config: Config, cases: list[Case], shown: Callable[[int, CaseResult], None]
) -> list[CaseResult]:
    results = []
    for number, case in enumerate(cases, start=1):
        results.append(run_case(make_agent(), config, case))
        shown(number, results[-1])
    return results


def save_results(path: Path, config: Config, results: list[CaseResult]) -> None:
    data = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.model,
        "judge_model": config.judge_model or config.model,
        "features": config.features,
        "summary": asdict(summarize(results)),
        "cases": [asdict(result) for result in results],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        raise EvalError(f"Cannot write the results {path}: {e}") from None


def load_results(path: Path) -> tuple[dict, list[CaseResult]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = [CaseResult(**case) for case in data["cases"]]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise EvalError(f"Cannot read the results {path}: {e}") from None
    return data, results
