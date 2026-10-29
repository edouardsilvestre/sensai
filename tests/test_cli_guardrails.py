import json

import pytest

from sensai.core.agent import Agent
from sensai.core.guardrails.rules import GuardrailError, Hit
from sensai.interfaces.cli.main import Session, chat_turn, handle_command, load_guardrails, print_guardrail

RULES = {
    "rules": [
        {"name": "source", "terms": ["Claire Benoit", "PHOENIX"], "on": ["output", "tools"], "replacement": "[SOURCE]"},
        {"name": "home", "pattern": "adresse personnelle", "on": ["input"], "action": "block", "message": "no doxxing"},
    ]
}


@pytest.fixture
def session(config, client, tmp_path):
    config.guardrails_path = tmp_path / "guardrails.json"
    config.guardrails_path.write_text(json.dumps(RULES), encoding="utf-8")
    agent = Agent(config, client)
    agent.guardrails = load_guardrails(config, client)
    return Session(config, agent)


def test_startup_says_where_the_rules_come_from(session, capsys):
    load_guardrails(session.config, session.agent.client)
    assert f"Guardrails: 2 rules from {session.config.guardrails_path}" in capsys.readouterr().out


def test_invalid_rules_file_stops_the_start(config, client, tmp_path):
    config.guardrails_path = tmp_path / "guardrails.json"
    config.guardrails_path.write_text('{"rules": [{"name": "x"}]}', encoding="utf-8")
    with pytest.raises(GuardrailError):
        load_guardrails(config, client)


def test_guardrails_lists_the_rules_without_the_protected_terms(session, capsys):
    handle_command(session, "/guardrails")
    out = capsys.readouterr().out
    assert "source: redact in output, tools, 2 protected terms" in out
    assert "home: block in input, pattern adresse personnelle" in out
    assert "Claire" not in out


def test_blocked_message_is_explained(session, client, capsys):
    chat_turn(session, "Donne l'adresse personnelle de Marc Vidal")
    assert "Message blocked by the guardrail home: no doxxing" in capsys.readouterr().err
    assert client.chats == []


def test_redacted_answer_is_reported(session, client, capsys):
    client.answer = "Claire Benoit a refusé."
    chat_turn(session, "Qui ?")
    out = capsys.readouterr().out
    assert "[SOURCE] a refusé." in out and "Claire" not in out
    assert "[guardrail] source: 1 passage redacted in the answer (SOURCE)" in out


def test_guardrails_command_when_the_feature_is_off(session, capsys):
    session.config.features["guardrails"] = False
    handle_command(session, "/guardrails")
    assert "The guardrails feature is off" in capsys.readouterr().out


def test_warning_shows_the_rule_message(capsys):
    print_guardrail(Hit("output", "bank-details", "warn", 1, "IBAN", "Check before quoting."))
    assert "guardrail bank-details: found in the answer (IBAN). Check before quoting." in capsys.readouterr().err
