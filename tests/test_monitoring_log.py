from datetime import datetime, timedelta, timezone

import pytest

from sensai.core.monitoring.log import Interaction, InteractionLog, LogError

NOW = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)
PROMPT = [
    {"role": "system", "content": "Tu es l'enquêteur."},
    {"role": "user", "content": "Qui dirige Orion ?"},
]


@pytest.fixture
def log(tmp_path):
    log = InteractionLog(tmp_path / "logs" / "logs.db")
    yield log
    log.close()


def interaction(**changes) -> Interaction:
    values = {"model": "qwen2.5:1.5b", "question": "Qui dirige Orion ?", "created_at": NOW}
    return Interaction(**(values | changes))


def test_an_interaction_is_read_back_as_it_was_added(log):
    added = interaction(
        session_id="20261101-100000", persona="investigator", prompt_version="3f2a1c9b", prompt=PROMPT,
        answer="Marc Vidal [1].", seconds=8.25, first_token_seconds=1.5, prompt_tokens=1200, completion_tokens=40,
        tools=["web_search", "memory_search"], sources=3, guardrails=["source"], reasoning_steps=2,
    )
    log.add(added)
    assert log.since(NOW - timedelta(days=1)) == [added]


def test_the_defaults_describe_a_plain_answer(log):
    log.add(interaction())
    [read] = log.since(NOW - timedelta(days=1))
    assert (read.outcome, read.error, read.first_token_seconds) == ("answered", "", None)
    assert (read.tools, read.guardrails, read.prompt) == ([], [], [])


def test_since_keeps_the_period_in_order(log):
    for days in (3, 0, 1):
        log.add(interaction(question=f"{days} days ago", created_at=NOW - timedelta(days=days)))
    assert [read.question for read in log.since(NOW - timedelta(days=2))] == ["1 days ago", "0 days ago"]


def test_since_compares_dates_in_any_time_zone(log):
    paris = timezone(timedelta(hours=1))
    log.add(interaction(created_at=datetime(2026, 11, 1, 11, 30, tzinfo=paris)))
    assert len(log.since(datetime(2026, 11, 1, 10, 15, tzinfo=timezone.utc))) == 1
    assert log.since(datetime(2026, 11, 1, 11, 45, tzinfo=paris)) == []


def test_the_log_is_kept_between_runs(tmp_path):
    path = tmp_path / "logs.db"
    first = InteractionLog(path)
    first.add(interaction(outcome="error", error="Cannot reach Ollama"))
    first.close()
    second = InteractionLog(path)
    assert [(read.outcome, read.error) for read in second.since(NOW - timedelta(days=1))] == [
        ("error", "Cannot reach Ollama")
    ]
    second.close()


def test_clear_removes_every_interaction(log):
    log.add(interaction())
    log.add(interaction(outcome="blocked"))
    assert log.clear() == 2
    assert log.since(NOW - timedelta(days=1)) == []


def test_an_unknown_outcome_is_refused(log):
    with pytest.raises(LogError, match="Unknown outcome 'done'"):
        log.add(interaction(outcome="done"))


def test_a_file_that_is_not_a_database_is_an_error(tmp_path):
    path = tmp_path / "logs.db"
    path.write_text("not a database", encoding="utf-8")
    with pytest.raises(LogError, match="Cannot open the log"):
        InteractionLog(path)


def test_a_folder_that_cannot_be_created_is_an_error(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("", encoding="utf-8")
    with pytest.raises(LogError, match="Cannot open the log"):
        InteractionLog(blocked / "logs.db")


def test_a_failed_write_is_an_error(log):
    log.close()
    with pytest.raises(LogError, match="Cannot write to the log"):
        log.add(interaction())
