import pytest

from conftest import FakeClient
from sensai.core.agent import INTERRUPTED, Agent
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.events import TokenEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.memory.sessions import SessionStore
from sensai.core.persona import load_personas
from sensai.core.structured import ANSWER_SCHEMA
from sensai.interfaces.cli.main import Session, chat_turn, handle_command, json_command


# Long enough for the guardrails filter, which holds back the last 80 characters, to show the beginning.
LONG_ANSWER = " ".join(f"word{number}" for number in range(60))


class InterruptedClient(FakeClient):
    """The user presses Ctrl+C after `words` words of the next answer."""

    def __init__(self, words: int):
        super().__init__(LONG_ANSWER)
        self.words = words

    def chat_stream(self, *args, **kwargs):
        stream = super().chat_stream(*args, **kwargs)
        for number, chunk in enumerate(stream):
            if self.words is not None and number == self.words:
                self.words = None
                raise KeyboardInterrupt
            yield chunk


def run_until_interrupted(agent: Agent, question: str, schema: dict | None = None) -> list:
    events = []
    with pytest.raises(KeyboardInterrupt):
        for event in agent.run(question, schema):
            events.append(event)
    return events


def shown(events: list) -> str:
    return "".join(event.text for event in events if isinstance(event, TokenEvent))


def test_the_interrupted_answer_stays_in_the_history(config):
    agent = Agent(config, InterruptedClient(30))
    events = run_until_interrupted(agent, "Who approved?")
    assert agent.interrupt()
    assert shown(events).startswith("word0 word1 ")
    assert agent.memory.turns[0].user == "Who approved?"
    assert agent.memory.turns[0].assistant == f"{shown(events)}\n{INTERRUPTED}"


def test_the_model_gets_exactly_what_the_user_saw(config):
    agent = Agent(config, InterruptedClient(3))
    events = run_until_interrupted(agent, "Who approved?")
    agent.interrupt()
    assert shown(events) == "" and agent.memory.turns[0].assistant == INTERRUPTED


def test_the_next_message_redirects_the_answer(config):
    client = InterruptedClient(30)
    agent = Agent(config, client)
    events = run_until_interrupted(agent, "Summarize the leak")
    agent.interrupt()
    list(agent.run("Shorter, and only the amounts"))
    assert client.chats[-1][0] == [
        {"role": "user", "content": "Summarize the leak"},
        {"role": "assistant", "content": f"{shown(events)}\n{INTERRUPTED}"},
        {"role": "user", "content": "Shorter, and only the amounts"},
    ]


def test_an_answer_interrupted_before_its_first_word_keeps_the_question(config):
    agent = Agent(config, InterruptedClient(0))
    run_until_interrupted(agent, "Who approved?")
    agent.interrupt()
    assert agent.memory.turns[0].assistant == INTERRUPTED


def test_only_what_the_guardrails_let_through_is_kept(config):
    client = InterruptedClient(30)
    client.answer = "It was Claire Benoit who talked " + LONG_ANSWER
    rule = {"name": "source", "terms": ["Claire Benoit"], "on": ["output"], "replacement": "[SOURCE]"}
    agent = Agent(config, client, guardrails=Guardrails(parse_rules([rule])))
    events = run_until_interrupted(agent, "Who talked?")
    agent.interrupt()
    assert agent.memory.turns[0].assistant == f"{shown(events)}\n{INTERRUPTED}"
    assert "[SOURCE]" in shown(events) and "Claire" not in agent.memory.turns[0].assistant


def test_an_interrupted_answer_does_not_update_the_document(config):
    client = InterruptedClient(1)
    client.edits = '{"edits": [{"op": "append", "section": "Summary", "text": "- half a fact"}]}'
    agent = Agent(config, client)
    run_until_interrupted(agent, "Who approved?")
    agent.interrupt()
    assert client.edit_requests == [] and agent.memory.document.is_empty


def test_a_json_answer_cut_halfway_is_not_kept(config):
    agent = Agent(config, InterruptedClient(1))
    run_until_interrupted(agent, "Who approved?", ANSWER_SCHEMA)
    assert not agent.interrupt() and agent.memory.turns == []


def test_nothing_to_keep_after_a_finished_answer(config, client):
    agent = Agent(config, client)
    list(agent.run("Who approved?"))
    assert not agent.interrupt()
    assert len(agent.memory.turns) == 1


def test_the_cli_keeps_the_answer_and_says_how_to_go_on(config, capsys):
    session = Session(config, Agent(config, InterruptedClient(30)))
    chat_turn(session, "Who approved?")
    out = capsys.readouterr().out
    assert "[interrupted] What was written is kept" in out
    assert session.agent.memory.turns[0].assistant.startswith("word0 word1 ")
    assert session.agent.memory.turns[0].assistant.endswith(INTERRUPTED)


def test_ctrl_c_while_the_cli_prints_keeps_the_answer_too(config, monkeypatch):
    session = Session(config, Agent(config, FakeClient(LONG_ANSWER)))
    printed = []

    def interrupted_print(*args, **kwargs):
        printed.append(args)
        if len(printed) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("builtins.print", interrupted_print)
    chat_turn(session, "Who approved?")
    monkeypatch.undo()
    assert session.agent.memory.turns[0].assistant == f"{printed[0][0]}{printed[1][0]}\n{INTERRUPTED}"


def test_with_the_feature_off_an_interrupted_turn_is_dropped(config, capsys):
    config.features["branching"] = False
    session = Session(config, Agent(config, InterruptedClient(1)))
    chat_turn(session, "Who approved?")
    assert session.agent.memory.turns == []
    assert "kept" not in capsys.readouterr().out


def test_the_interrupted_turn_is_saved(config, tmp_path):
    store = SessionStore(tmp_path / "saved")
    session = Session(config, Agent(config, InterruptedClient(1)), store=store)
    chat_turn(session, "Who approved?")
    assert store.list()[0].turns == 1


def test_an_interrupted_review_leaves_the_conversation_as_it_was(config, capsys):
    session = Session(config, Agent(config, InterruptedClient(1)), personas=load_personas(DEFAULT_PERSONAS_DIR))
    handle_command(session, "/review Who approved?")
    assert session.agent.memory.turns == []
    assert "[interrupted]" in capsys.readouterr().out


def test_an_interrupted_json_command_keeps_nothing(config, capsys):
    session = Session(config, Agent(config, InterruptedClient(1)))
    json_command(session, "Who approved?")
    assert session.agent.memory.turns == []
    assert "[interrupted]\n" in capsys.readouterr().out
