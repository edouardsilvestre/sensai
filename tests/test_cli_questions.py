import pytest

from sensai.core.agent import Agent
from sensai.core.llm.ollama import ToolCall
from sensai.interfaces.cli.main import Session, ask_question, chat_turn, load_memory, load_tools

QUESTION = "Plusieurs sociétés Orion apparaissent dans les documents. Laquelle ?"
OPTIONS = ["Orion Delta Consulting (Lyon)", "Orion Holding Ltd (Chypre)"]


def answer(monkeypatch, *answers):
    replies = iter(answers)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        reply = next(replies)
        if reply is EOFError:
            raise EOFError
        return reply

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def test_the_question_and_its_options_are_shown(monkeypatch, capsys):
    prompts = answer(monkeypatch, "1")
    ask_question(QUESTION, OPTIONS)
    assert capsys.readouterr().out.splitlines() == [
        f"[ask_user] {QUESTION}",
        "  1. Orion Delta Consulting (Lyon)",
        "  2. Orion Holding Ltd (Chypre)",
    ]
    assert prompts == ["Your answer (a number or your own words): "]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("2", "Orion Holding Ltd (Chypre)"), (" 1 ", "Orion Delta Consulting (Lyon)"),
        ("les deux", "les deux"), ("3", "3"),
    ],
)
def test_a_number_picks_an_option_and_anything_else_is_sent_as_it_is(monkeypatch, reply, expected):
    answer(monkeypatch, reply)
    assert ask_question(QUESTION, OPTIONS) == expected


def test_without_options_the_answer_is_free(monkeypatch, capsys):
    prompts = answer(monkeypatch, "1")
    assert ask_question("Quelle période ?", []) == "1"
    assert capsys.readouterr().out == "[ask_user] Quelle période ?\n"
    assert prompts == ["Your answer: "]


def test_an_empty_answer_asks_again(monkeypatch):
    prompts = answer(monkeypatch, "", "   ", "Lyon")
    assert ask_question(QUESTION, OPTIONS) == "Lyon"
    assert len(prompts) == 3


def test_ctrl_d_gives_no_answer(monkeypatch):
    answer(monkeypatch, EOFError)
    assert ask_question(QUESTION, OPTIONS) is None


def test_the_cli_gives_ask_user_to_the_model(config, client):
    assert "ask_user" in load_tools(config, client).names()
    config.features["questions"] = False
    assert "ask_user" not in load_tools(config, client).names()


def test_the_model_asks_and_gets_the_answer(config, client, monkeypatch, capsys):
    agent = Agent(config, client)
    agent.tools = load_tools(config, client)
    session = Session(config, agent)
    load_memory(session, None)
    client.tool_rounds = [[ToolCall("ask_user", {"question": QUESTION, "options": OPTIONS})]]
    answer(monkeypatch, "1")

    chat_turn(session, "Trouve les virements d'Orion vers l'étranger.")

    out = capsys.readouterr().out
    assert out.count(QUESTION) == 1
    assert '[ask_user] answered "Orion Delta Consulting (Lyon)"' in out
    assert client.chats[1][0][-1]["content"] == "The user answered: Orion Delta Consulting (Lyon)"
