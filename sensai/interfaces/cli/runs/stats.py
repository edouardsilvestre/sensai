from datetime import datetime, timedelta, timezone

from sensai.core.config import Config
from sensai.core.monitoring.log import InteractionLog, LogError
from sensai.core.monitoring.metrics import Group, Report, Timing, summarize
from sensai.interfaces.cli.output import error, plural, seconds

OUTCOME_LABELS = {
    "answered": "answered", "cached": "from the cache", "blocked": "blocked", "stopped": "stopped", "error": "errors",
}


def describe_timing(timing: Timing | None) -> str:
    if timing is None:
        return "no answer"
    return f"{plural(timing.answers, 'answer')}, {seconds(timing.average)}, p95 {seconds(timing.p95)}"


def print_groups(title: str, groups: list[Group], label=lambda group: group.name) -> None:
    print(f"\n{title}:")
    width = max(len(label(group)) for group in groups)
    for group in groups:
        errors = f", {plural(group.errors, 'error')}" if group.errors else ""
        messages = plural(group.messages, "message")
        print(f"  {label(group):<{width}}  {messages}{errors}: {describe_timing(group.timing)}")


def print_stats(report: Report, days: int) -> None:
    if not report.messages:
        print(f"No messages logged in the last {plural(days, 'day')}.")
        return
    print(
        f"Last {plural(days, 'day')}: {plural(report.messages, 'message')} in "
        f"{plural(report.conversations, 'conversation')}, {plural(report.errors, 'error')} ({report.error_rate:.0%})."
    )
    timing = report.timing
    if timing is not None:
        first = f", first word after {seconds(timing.first_word)}" if timing.first_word is not None else ""
        print(
            f"Answers: {seconds(timing.average)} on average, {seconds(timing.p95)} for the slowest 5% (p95){first}."
        )
        print(f"Tokens: {report.prompt_tokens} sent and {report.completion_tokens} received per answer.")
    outcomes = [f"{count} {OUTCOME_LABELS[outcome]}" for outcome, count in report.outcomes.items() if count]
    print(f"Outcomes: {', '.join(outcomes)}.")
    print_groups("By day", report.days)
    drifts = {drift.version: drift for drift in report.drifts}
    print("\nBy prompt version:")
    for group in report.versions:
        drift = drifts.get(group.name)
        alert = f"  [drift] {drift.change:+.0%} slower than {drift.previous} on {drift.model}" if drift else ""
        since = f"{group.since.astimezone():%Y-%m-%d %H:%M}"
        print(f"  {group.name or '-'}  since {since}: {describe_timing(group.timing)}{alert}")
    print_groups("By model", report.models)
    print_groups("By persona", report.personas, lambda group: group.name or "no persona")
    if report.tools:
        print(f"\nTools: {', '.join(f'{name} {count}' for name, count in report.tools)}.")


def logged_since(log: InteractionLog, days: int) -> Report:
    return summarize(log.since(datetime.now(timezone.utc) - timedelta(days=days)))


def run_stats(config: Config, days: int) -> int:
    if days < 1:
        error("--days must be at least 1")
        return 1
    if not config.log_path.exists():
        print(f"No log at {config.log_path} yet: chat with Sensai, every message is logged.")
        return 0
    try:
        log = InteractionLog(config.log_path)
    except LogError as e:
        error(str(e))
        return 1
    try:
        print_stats(logged_since(log, days), days)
    except LogError as e:
        error(str(e))
        return 1
    finally:
        log.close()
    return 0
