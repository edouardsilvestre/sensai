import time
from datetime import datetime, timedelta, timezone

import pytest

from sensai.core.monitoring.log import Interaction
from sensai.core.monitoring.metrics import Drift, summarize

DAY = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def paris(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/Paris")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def interaction(seconds: float = 5.0, minutes: int = 0, **changes) -> Interaction:
    values = {
        "model": "qwen2.5:1.5b", "question": "Qui dirige Orion ?", "created_at": DAY + timedelta(minutes=minutes),
        "seconds": seconds, "prompt_version": "aaaaaaaa", "session_id": "20261030-120000",
    }
    return Interaction(**(values | changes))


def test_an_empty_log_has_no_metrics():
    report = summarize([])
    assert (report.messages, report.errors, report.error_rate, report.timing) == (0, 0, 0.0, None)


def test_the_times_are_measured_on_the_answers_of_the_model_only():
    report = summarize([
        interaction(4.0, first_token_seconds=1.0),
        interaction(8.0, first_token_seconds=3.0, outcome="stopped"),
        interaction(0.5, outcome="cached"),
        interaction(0.1, outcome="blocked"),
        interaction(30.0, outcome="error", error="Cannot reach Ollama"),
    ])
    assert (report.timing.answers, report.timing.average, report.timing.first_word) == (2, 6.0, 2.0)
    assert report.outcomes == {"answered": 1, "cached": 1, "blocked": 1, "stopped": 1, "error": 1}
    assert (report.messages, report.errors, report.error_rate) == (5, 1, 0.2)


def test_p95_is_the_time_95_percent_of_the_answers_stay_under():
    report = summarize([interaction(float(seconds)) for seconds in range(1, 21)])
    assert (report.timing.p95, report.timing.average) == (19.0, 10.5)
    assert summarize([interaction(3.0)]).timing.p95 == 3.0


def test_tokens_are_averaged_per_answer():
    report = summarize([
        interaction(prompt_tokens=1000, completion_tokens=50),
        interaction(prompt_tokens=1400, completion_tokens=110),
        interaction(outcome="cached"),
    ])
    assert (report.prompt_tokens, report.completion_tokens) == (1200, 80)


def test_conversations_are_counted_once():
    other, unsaved = interaction(session_id="20261030-130000"), interaction(session_id="")
    report = summarize([interaction(), other, interaction(), unsaved])
    assert report.conversations == 2


def test_messages_are_grouped_by_local_day():
    report = summarize([
        interaction(minutes=0),
        interaction(minutes=11 * 60 + 30),
        interaction(minutes=24 * 60, outcome="error"),
    ])
    assert [(day.name, day.messages, day.errors) for day in report.days] == [
        ("2026-10-30", 1, 0), ("2026-10-31", 2, 1)
    ]


def test_groups_by_version_model_and_persona_follow_their_first_message():
    report = summarize([
        interaction(minutes=5, prompt_version="bbbbbbbb", persona="investigator"),
        interaction(minutes=1, model="llama3"),
        interaction(minutes=9, persona="investigator"),
    ])
    assert [(group.name, group.messages) for group in report.versions] == [("aaaaaaaa", 2), ("bbbbbbbb", 1)]
    assert [group.name for group in report.models] == ["llama3", "qwen2.5:1.5b"]
    assert [group.name for group in report.personas] == ["", "investigator"]
    assert report.versions[1].since == DAY + timedelta(minutes=5)


def test_the_tools_are_counted_from_the_most_called():
    report = summarize([interaction(tools=["lookup", "web_search"]), interaction(tools=["web_search"])])
    assert report.tools == [("web_search", 2), ("lookup", 1)]


def test_a_slower_prompt_version_is_a_drift():
    before = [interaction(4.0, minutes=minute) for minute in range(3)]
    after = [interaction(9.0, minutes=60 + minute, prompt_version="bbbbbbbb") for minute in range(3)]
    report = summarize(before + after)
    assert report.drifts == [Drift("qwen2.5:1.5b", "bbbbbbbb", "aaaaaaaa", 1.25)]


@pytest.mark.parametrize(
    "after",
    [
        [interaction(5.5, minutes=60 + minute, prompt_version="bbbbbbbb") for minute in range(3)],
        [interaction(9.0, minutes=60 + minute, prompt_version="bbbbbbbb") for minute in range(2)],
        [interaction(9.0, minutes=60 + minute, prompt_version="bbbbbbbb", model="llama3") for minute in range(3)],
    ],
    ids=["a little slower", "too few answers", "another model"],
)
def test_no_drift(after):
    before = [interaction(4.0, minutes=minute) for minute in range(3)]
    assert summarize(before + after).drifts == []
