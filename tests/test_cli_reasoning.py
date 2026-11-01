import json

import pytest

from sensai.core.agent import Agent
from sensai.core.memory.sessions import SessionStore
from sensai.core.reasoning import ANSWER
from sensai.core.tools.registry import ToolRegistry
from sensai.interfaces.cli.main import Session, chat_turn, handle_command, load_memory

FOUND = "statuts_orion.pdf: Marc Vidal is the manager of Orion Delta Consulting."


def search_documents(query: str) -> str:
    """
    Search the journalist's indexed documents.

    Args:
      query: What to look for

    Returns:
      The excerpts found
    """
    return FOUND


def step(thought: str, action: str = ANSWER, **arguments) -> str:
    return json.dumps({"thought": thought, "action": action, "arguments": arguments})


@pytest.fixture
def session(config, client):
    config.features["reasoning"] = True
    session = Session(config, Agent(config, client, tools=ToolRegistry([search_documents])))
    load_memory(session, None)
    return session


def ask(session, client, question, *steps):
    client.steps = list(steps)
    chat_turn(session, question)


def test_the_steps_are_shown_before_the_answer(session, client, capsys):
    client.answer = "Marc Vidal dirige Orion."
    ask(session, client, "Qui dirige Orion ?", step("I need the manager.", "search_documents", query="Orion"),
        step("Marc Vidal manages Orion."))
    out = capsys.readouterr().out
    assert out.index("[thought 1] I need the manager.") < out.index('[search_documents] query="Orion"')
    assert out.index('[search_documents] query="Orion"') < out.index("[thought 2] Marc Vidal manages Orion.")
    assert out.index("[thought 2]") < out.index("Marc Vidal dirige Orion.")


def test_a_stopped_reasoning_is_reported(session, client, capsys):
    ask(session, client, "Qui dirige Orion ?", "not a step")
    assert "[warning] Reasoning stopped: The step is not valid JSON" in capsys.readouterr().err


def test_reasoning_without_history(session, capsys):
    handle_command(session, "/reasoning")
    assert "No reasoning yet in this conversation: ask a question." in capsys.readouterr().out
    handle_command(session, "/feature reasoning off")
    handle_command(session, "/reasoning")
    assert "No reasoning yet in this conversation: /feature reasoning on, then ask one." in capsys.readouterr().out


def test_reasoning_lists_the_questions_of_the_conversation(session, client, capsys):
    ask(session, client, "Qui dirige Orion ?", step("I need the manager.", "search_documents", query="Orion"),
        step("Marc Vidal."))
    ask(session, client, "Merci", step("Nothing to look up."))
    capsys.readouterr()
    handle_command(session, "/reasoning")
    assert capsys.readouterr().out.splitlines() == [
        "Reasoning of this conversation:",
        "  1. Qui dirige Orion ? (2 steps)",
        "  2. Merci (1 step)",
        "/reasoning N to read the steps of question N.",
    ]


def test_reasoning_shows_the_steps_and_the_results_of_a_question(session, client, capsys):
    ask(session, client, "Qui dirige Orion ?", step("I need the manager.", "search_documents", query="Orion"),
        step("Marc Vidal."))
    capsys.readouterr()
    handle_command(session, "/reasoning 1")
    assert capsys.readouterr().out.splitlines() == [
        "Question 1: Qui dirige Orion ?",
        "[thought 1] I need the manager.",
        '  [search_documents] query="Orion"',
        f"    {FOUND}",
        "[thought 2] Marc Vidal.",
    ]


@pytest.mark.parametrize("argument", ["2", "0", "abc", "²"])
def test_reasoning_refuses_an_unknown_question(session, client, capsys, argument):
    ask(session, client, "Merci", step("Nothing to look up."))
    handle_command(session, f"/reasoning {argument}")
    assert "Usage: /reasoning [N], with N between 1 and 1" in capsys.readouterr().err


def test_the_history_stays_readable_once_the_feature_is_off(session, client, capsys):
    ask(session, client, "Merci", step("Nothing to look up."))
    handle_command(session, "/feature reasoning off")
    capsys.readouterr()
    handle_command(session, "/reasoning")
    assert "  1. Merci (1 step)" in capsys.readouterr().out


def test_a_new_conversation_forgets_the_reasoning(session, client, capsys):
    ask(session, client, "Merci", step("Nothing to look up."))
    handle_command(session, "/new")
    assert session.agent.reasoning == []


def test_resuming_a_conversation_forgets_the_reasoning(session, client, capsys):
    SessionStore(session.config.sessions_dir).save("alvor", session.agent.memory, "m")
    ask(session, client, "Merci", step("Nothing to look up."))
    handle_command(session, "/resume alvor")
    assert session.agent.reasoning == []


def test_help_lists_the_reasoning_command(session, capsys):
    handle_command(session, "/help")
    assert "/reasoning [N]" in capsys.readouterr().out
