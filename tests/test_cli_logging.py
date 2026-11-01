from datetime import datetime, timedelta, timezone

import pytest

from sensai.core.agent import Agent
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.monitoring.log import InteractionLog
from sensai.core.persona import load_personas
from sensai.interfaces.cli.answer import chat_turn
from sensai.interfaces.cli.commands.dispatch import handle_command
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import load_log, load_memory

ANSWER = "Marc Vidal dirige Orion."
APPROVED = '{"claims": [{"claim": "Marc Vidal runs Orion", "verdict": "supported", "reason": "the statutes say so"}]}'


@pytest.fixture
def session(config, client):
    config.features["document"] = False
    client.answer = ANSWER
    agent = Agent(config, client)
    session = Session(config, agent)
    session.log = agent.log = load_log(config)
    load_memory(session, None)
    yield session
    if session.log is not None:
        session.log.close()


def logged(session) -> list:
    return session.log.since(datetime.now(timezone.utc) - timedelta(minutes=1))


def test_the_first_message_is_logged_with_the_id_its_conversation_is_saved_under(session):
    chat_turn(session, "Qui dirige Orion ?")
    chat_turn(session, "Depuis quand ?")
    assert session.session_id is not None
    assert [record.session_id for record in logged(session)] == [session.session_id] * 2
    assert session.store.load(session.session_id, session.agent.memory) == 2


def test_a_new_conversation_gets_a_new_id(session, monkeypatch):
    ids = iter(["20261101-100000", "20261101-100500"])
    monkeypatch.setattr(session.store, "new_id", lambda: next(ids))
    chat_turn(session, "Qui dirige Orion ?")
    handle_command(session, "/new")
    chat_turn(session, "Où est le siège d'Orion ?")
    assert [record.session_id for record in logged(session)] == ["20261101-100000", "20261101-100500"]


def test_a_resumed_conversation_keeps_its_id(session, capsys):
    chat_turn(session, "Qui dirige Orion ?")
    first = session.session_id
    handle_command(session, "/new")
    handle_command(session, f"/resume {first}")
    chat_turn(session, "Depuis quand ?")
    assert [record.session_id for record in logged(session)] == [first, first]


def test_a_first_review_is_logged_with_the_conversation_id(session, client):
    answers = iter([ANSWER, APPROVED, ANSWER])
    original = client.chat_stream

    def scripted(*args, **kwargs):
        client.answer = next(answers)
        return original(*args, **kwargs)

    client.chat_stream = scripted
    session.personas = load_personas(DEFAULT_PERSONAS_DIR)
    handle_command(session, "/review Qui dirige Orion ?")
    records = logged(session)
    assert [record.persona for record in records] == ["investigator", "fact_checker", "editor"]
    assert {record.session_id for record in records} == {session.session_id}


def test_a_log_that_cannot_be_written_is_a_warning(session, capsys):
    session.log.close()
    chat_turn(session, "Qui dirige Orion ?")
    out, err = capsys.readouterr()
    assert "[warning] This message was not logged: Cannot write to the log" in err
    assert ANSWER in out
    session.log = None


def test_an_unreadable_log_disables_the_logging(config, capsys):
    config.log_path.write_text("not a database", encoding="utf-8")
    assert load_log(config) is None
    assert "[warning] Logging disabled: Cannot open the log" in capsys.readouterr().err


def test_feature_logging_on_opens_the_log(config, client, capsys):
    config.features["logging"] = False
    session = Session(config, Agent(config, client))
    handle_command(session, "/feature logging on")
    assert isinstance(session.log, InteractionLog) and session.agent.log is session.log
    session.log.close()
