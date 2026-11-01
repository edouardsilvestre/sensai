import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from sensai.core.monitoring.log import OUTCOMES, Interaction

ANSWERED = ("answered", "stopped")
DRIFT = 0.5
DRIFT_MIN_ANSWERS = 3


@dataclass
class Timing:
    answers: int
    average: float
    p95: float
    first_word: float | None


@dataclass
class Group:
    name: str
    messages: int
    errors: int
    since: datetime
    timing: Timing | None


@dataclass
class Drift:
    model: str
    version: str
    previous: str
    change: float


@dataclass
class Report:
    messages: int
    conversations: int
    outcomes: dict[str, int]
    timing: Timing | None
    prompt_tokens: int
    completion_tokens: int
    days: list[Group] = field(default_factory=list)
    versions: list[Group] = field(default_factory=list)
    models: list[Group] = field(default_factory=list)
    personas: list[Group] = field(default_factory=list)
    tools: list[tuple[str, int]] = field(default_factory=list)
    drifts: list[Drift] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.outcomes["error"]

    @property
    def error_rate(self) -> float:
        return self.errors / self.messages if self.messages else 0.0


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def timing(interactions: list[Interaction]) -> Timing | None:
    answered = [interaction for interaction in interactions if interaction.outcome in ANSWERED]
    if not answered:
        return None
    seconds = [interaction.seconds for interaction in answered]
    first = [interaction.first_token_seconds for interaction in answered if interaction.first_token_seconds is not None]
    return Timing(len(answered), sum(seconds) / len(seconds), _p95(seconds), sum(first) / len(first) if first else None)


def _groups(interactions: list[Interaction], key: Callable[[Interaction], str]) -> list[Group]:
    grouped: dict[str, list[Interaction]] = {}
    for interaction in sorted(interactions, key=lambda interaction: interaction.created_at):
        grouped.setdefault(key(interaction), []).append(interaction)
    return [
        Group(
            name, len(members), sum(member.outcome == "error" for member in members), members[0].created_at,
            timing(members),
        )
        for name, members in grouped.items()
    ]


def _drifts(interactions: list[Interaction]) -> list[Drift]:
    drifts = []
    for model in _groups(interactions, lambda interaction: interaction.model):
        versions = _groups(
            [interaction for interaction in interactions if interaction.model == model.name],
            lambda interaction: interaction.prompt_version,
        )
        measured = [version for version in versions if version.timing and version.timing.answers >= DRIFT_MIN_ANSWERS]
        for previous, version in zip(measured, measured[1:]):
            change = version.timing.average / previous.timing.average - 1 if previous.timing.average else 0.0
            if change >= DRIFT:
                drifts.append(Drift(model.name, version.name, previous.name, change))
    return drifts


def summarize(interactions: list[Interaction]) -> Report:
    outcomes = Counter(interaction.outcome for interaction in interactions)
    answered = [interaction for interaction in interactions if interaction.outcome in ANSWERED]
    tools = Counter(tool for interaction in interactions for tool in interaction.tools)
    return Report(
        messages=len(interactions),
        conversations=len({interaction.session_id for interaction in interactions if interaction.session_id}),
        outcomes={outcome: outcomes[outcome] for outcome in OUTCOMES},
        timing=timing(interactions),
        prompt_tokens=round(sum(i.prompt_tokens for i in answered) / len(answered)) if answered else 0,
        completion_tokens=round(sum(i.completion_tokens for i in answered) / len(answered)) if answered else 0,
        days=_groups(interactions, lambda interaction: interaction.created_at.astimezone().strftime("%Y-%m-%d")),
        versions=_groups(interactions, lambda interaction: interaction.prompt_version),
        models=_groups(interactions, lambda interaction: interaction.model),
        personas=_groups(interactions, lambda interaction: interaction.persona),
        tools=tools.most_common(),
        drifts=_drifts(interactions),
    )
