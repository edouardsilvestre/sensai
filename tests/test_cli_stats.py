from datetime import datetime, timedelta, timezone

import pytest

from sensai.core.agent import Agent
from sensai.core.monitoring.log import Interaction, InteractionLog
from sensai.interfaces.cli.main import HELP, Session, chat_turn, handle_command, load_log, load_memory, main

NOW = datetime.now(timezone.utc)


def interaction(days: float = 0, **changes) -> Interaction:
    values = {
        "model": "qwen2.5:1.5b", "question": "Qui dirige Orion ?", "created_at": NOW - timedelta(days=days),
        "seconds": 4.0, "first_token_seconds": 1.0, "prompt_tokens": 1200, "completion_tokens": 80,
        "prompt_version": "aaaaaaaa", "session_id": "20261101-100000",
    }
    return Interaction(**(values | changes))


@pytest.fixture
def session(config, client):
    config.features["document"] = False
    agent = Agent(config, client)
    session = Session(config, agent)
    session.log = agent.log = load_log(config)
    yield session
    if session.log is not None:
        session.log.close()


def fill(log: InteractionLog) -> None:
    for minute in range(3):
        log.add(interaction(days=2, created_at=NOW - timedelta(days=2, minutes=minute)))
    for minute in range(3):
        log.add(interaction(seconds=9.0, prompt_version="bbbbbbbb", created_at=NOW - timedelta(minutes=minute)))
    log.add(interaction(outcome="error", error="Cannot reach Ollama", tools=["web_search"], persona="investigator"))
    log.add(interaction(days=20))


def test_stats_shows_the_metrics_of_the_last_week(session, capsys):
    fill(session.log)
    handle_command(session, "/stats")
    out = capsys.readouterr().out
    assert "Last 7 days: 7 messages in 1 conversation, 1 error (14%)." in out
    assert "Answers: 6.5 s on average, 9.0 s for the slowest 5% (p95), first word after 1.0 s." in out
    assert "Tokens: 1200 sent and 80 received per answer." in out
    assert "Outcomes: 6 answered, 1 errors." in out
    assert "bbbbbbbb  since " in out and "[drift] +125% slower than aaaaaaaa on qwen2.5:1.5b" in out
    assert "  no persona    6 messages: 6 answers, 6.5 s, p95 9.0 s" in out
    assert "  investigator  1 message, 1 error: no answer" in out
    assert "Tools: web_search 1." in out


def test_stats_takes_a_number_of_days(session, capsys):
    fill(session.log)
    handle_command(session, "/stats 30")
    assert "Last 30 days: 8 messages" in capsys.readouterr().out


def test_stats_of_a_live_conversation(session, capsys):
    load_memory(session, None)
    chat_turn(session, "Qui dirige Orion ?")
    capsys.readouterr()
    handle_command(session, "/stats 1")
    out = capsys.readouterr().out
    assert "Last 1 day: 1 message in 1 conversation, 0 errors (0%)." in out
    assert "Tokens: 100 sent and 10 received per answer." in out


def test_stats_without_messages(session, capsys):
    handle_command(session, "/stats")
    assert capsys.readouterr().out == "No messages logged in the last 7 days.\n"


def test_stats_clear_empties_the_log(session, capsys):
    fill(session.log)
    handle_command(session, "/stats clear")
    assert capsys.readouterr().out == "Log cleared: 8 messages removed.\n"
    assert session.log.since(NOW - timedelta(days=365)) == []


@pytest.mark.parametrize("argument", ["0", "week", "-3", "²"])
def test_stats_rejects_other_arguments(session, capsys, argument):
    handle_command(session, f"/stats {argument}")
    assert "Usage: /stats [DAYS|clear]" in capsys.readouterr().err


def test_stats_without_a_log(session, config, capsys):
    session.log.close()
    session.log = None
    handle_command(session, "/stats")
    assert capsys.readouterr().out == "The log could not be opened: /feature logging on to try again.\n"
    config.features["logging"] = False
    handle_command(session, "/stats")
    assert "The logging feature is off" in capsys.readouterr().out


@pytest.fixture
def env_file(tmp_path, config, monkeypatch) -> str:
    monkeypatch.setenv("SENSAI_LOG_PATH", str(config.log_path))
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    return str(env)


def test_the_stats_command_reads_the_log_without_ollama(env_file, config, capsys):
    log = InteractionLog(config.log_path)
    fill(log)
    log.close()
    assert main(["-e", env_file, "stats", "--days", "3"]) == 0
    assert "Last 3 days: 7 messages" in capsys.readouterr().out


def test_the_stats_command_without_a_log(env_file, config, capsys):
    assert main(["-e", env_file, "stats"]) == 0
    assert f"No log at {config.log_path} yet" in capsys.readouterr().out


def test_the_stats_command_rejects_zero_days(env_file, capsys):
    assert main(["-e", env_file, "stats", "--days", "0"]) == 1
    assert "--days must be at least 1" in capsys.readouterr().err


def test_help_lists_the_stats_command():
    assert "/stats [DAYS|clear]" in HELP
