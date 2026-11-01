import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sensai.core.agent import Agent, EmptyInputError
from sensai.core.config import Config
from sensai.core.events import DoneEvent, GuardrailEvent
from sensai.core.rag.models import SearchResult
from sensai.core.adversarial.models import (
    CATEGORIES,
    AdversarialError,
    AttackCase,
    AttackResult,
)


class AdversarialRetriever:
    def __init__(self, text: str):
        self.text = text

    def search(self, query: str, k: int) -> list[SearchResult]:
        return [SearchResult(1, self.text, "p. 1", "adversarial_doc.pdf", "", 0.95)]


@dataclass
class CategorySummary:
    total: int = 0
    defended: int = 0
    breached: int = 0
    errored: int = 0

    @property
    def defense_rate(self) -> float:
        return self.defended / self.total if self.total else 0.0


@dataclass
class AttackSummary:
    total: int = 0
    defended: int = 0
    breached: int = 0
    errored: int = 0
    defense_rate: float = 0.0
    by_category: dict[str, CategorySummary] = field(default_factory=dict)
    average_duration: float = 0.0


def run_attack_case(agent: Agent, case: AttackCase) -> AttackResult:
    start = time.monotonic()
    original_retriever = agent.retriever
    if case.document is not None:
        agent.retriever = AdversarialRetriever(case.document)

    blocked = False
    output = ""
    error = ""
    try:
        for event in agent.run(case.prompt):
            if isinstance(event, GuardrailEvent) and event.hit.action == "block":
                blocked = True
            elif isinstance(event, DoneEvent):
                output = event.content.strip()
    except EmptyInputError:
        output = ""
        error = ""
    except Exception as e:
        error = str(e)
    finally:
        agent.retriever = original_retriever

    duration = round(time.monotonic() - start, 2)
    status, reason = case.oracle.evaluate(blocked=blocked, output=output, error=error)
    return AttackResult(
        case=case,
        status=status,
        reason=reason,
        output=output,
        duration=duration,
    )


def summarize_attacks(results: list[AttackResult]) -> AttackSummary:
    by_category: dict[str, CategorySummary] = {cat: CategorySummary() for cat in CATEGORIES}
    total = len(results)
    defended = sum(r.status == "defended" for r in results)
    breached = sum(r.status == "breached" for r in results)
    errored = sum(r.status == "errored" for r in results)
    defense_rate = defended / total if total else 0.0
    total_duration = sum(r.duration for r in results)
    avg_duration = round(total_duration / total, 2) if total else 0.0

    for r in results:
        cat = r.case.category
        if cat not in by_category:
            by_category[cat] = CategorySummary()
        summary = by_category[cat]
        summary.total += 1
        if r.status == "defended":
            summary.defended += 1
        elif r.status == "breached":
            summary.breached += 1
        elif r.status == "errored":
            summary.errored += 1

    return AttackSummary(
        total=total,
        defended=defended,
        breached=breached,
        errored=errored,
        defense_rate=defense_rate,
        by_category=by_category,
        average_duration=avg_duration,
    )


def run_attack_suite(
    make_agent: Callable[[], Agent],
    cases: list[AttackCase],
    on_case: Callable[[int, AttackResult], None] | None = None,
) -> list[AttackResult]:
    results = []
    for number, case in enumerate(cases, start=1):
        agent = make_agent()
        result = run_attack_case(agent, case)
        results.append(result)
        if on_case is not None:
            on_case(number, result)
    return results


def save_attack_report(path: Path, config: Config, results: list[AttackResult]) -> None:
    summary = summarize_attacks(results)
    data = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.model,
        "features": config.features,
        "summary": {
            "total": summary.total,
            "defended": summary.defended,
            "breached": summary.breached,
            "errored": summary.errored,
            "defense_rate": summary.defense_rate,
            "average_duration": summary.average_duration,
            "by_category": {
                cat: {
                    "total": cat_sum.total,
                    "defended": cat_sum.defended,
                    "breached": cat_sum.breached,
                    "errored": cat_sum.errored,
                    "defense_rate": cat_sum.defense_rate,
                }
                for cat, cat_sum in summary.by_category.items()
            },
        },
        "results": [
            {
                "id": r.case.id,
                "category": r.case.category,
                "name": r.case.name,
                "prompt": r.case.prompt,
                "document": r.case.document,
                "status": r.status,
                "reason": r.reason,
                "output": r.output,
                "duration": r.duration,
            }
            for r in results
        ],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        raise AdversarialError(f"Cannot save attack report to {path}: {e}") from None


def load_attack_report(path: Path) -> tuple[dict, list[dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AdversarialError(f"Cannot read attack report from {path}: {e}") from None
    if not isinstance(data, dict) or "results" not in data or "summary" not in data:
        raise AdversarialError(f"Invalid attack report format in {path}")
    return data, data["results"]
