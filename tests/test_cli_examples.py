import json

import pytest

from sensai.core.agent import Agent
from sensai.core.examples import Example, ExampleLibrary, Excerpt
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ModelNotFoundError
from sensai.core.rag.models import SearchResult
from sensai.interfaces.cli.main import HELP, Session, chat_turn, handle_command, load_examples, load_memory

SALARY = "Quel salaire touche Marc Vidal ?"
BONUS = "Quel bonus touche Marc Vidal ?"
ANSWER = "Les documents ne mentionnent pas de salaire. Marc Vidal est gérant [2]."
STATUTES = SearchResult(7, "Gérant : Marc Vidal.", "p. 2", "statuts_orion.pdf", "", 0.9)
STATEMENT = SearchResult(8, "05/04/2023 virement Nordbank, 480 000 EUR.", "row 12", "releve.xlsx", "", 0.8)


class Retriever:
    def search(self, query: str, k: int) -> list[SearchResult]:
        return [STATEMENT, STATUTES]


@pytest.fixture
def session(config, client):
    config.features["document"] = False
    client.answer = ANSWER
    agent = Agent(config, client)
    session = Session(config, agent)
    session.examples = agent.examples = load_examples(config, client)
    return session


def kept(session) -> list[Example]:
    return ExampleLibrary(session.config.examples_path, lambda text: [1.0]).examples


def test_add_keeps_the_last_answer_with_its_cited_excerpts(session, capsys):
    session.agent.retriever = Retriever()
    chat_turn(session, SALARY)
    capsys.readouterr()

    handle_command(session, "/example add")

    assert capsys.readouterr().out == f'Answer kept as example 1: "{SALARY}"\n'
    [example] = kept(session)
    assert (example.question, example.answer) == (SALARY, ANSWER)
    assert example.excerpts == [Excerpt(2, "statuts_orion.pdf", "p. 2", "Gérant : Marc Vidal.")]


def test_a_kept_answer_is_shown_to_the_model_for_a_close_question(session, client, capsys):
    chat_turn(session, SALARY)
    handle_command(session, "/example add")
    capsys.readouterr()

    chat_turn(session, BONUS)

    assert f'[examples] 1 example: "{SALARY}"' in capsys.readouterr().out
    assert f"Message: {SALARY}\nAnswer: {ANSWER}" in client.chats[-1][0][-2]["content"]


def test_add_without_an_answer_is_refused(session, capsys):
    handle_command(session, "/example add")
    assert "No answer to keep yet" in capsys.readouterr().err
    assert kept(session) == []


def test_add_after_new_is_refused(session, capsys):
    chat_turn(session, SALARY)
    handle_command(session, "/new")
    handle_command(session, "/example add")
    assert "No answer to keep yet" in capsys.readouterr().err


def test_a_json_answer_cannot_be_kept(session, client, capsys):
    chat_turn(session, SALARY)
    client.answer = '{"answer": "Marc Vidal."}'
    handle_command(session, f"/json {BONUS}")
    handle_command(session, "/example add")
    assert "No answer to keep yet" in capsys.readouterr().err


def test_an_answer_stopped_by_a_guardrail_cannot_be_kept(session, client, capsys):
    rule = {"name": "iban", "detectors": ["iban"], "action": "block"}
    session.agent.guardrails = Guardrails(parse_rules([rule]))
    client.answer = "Le compte FR7630006000011234567890189 a reçu les fonds."
    chat_turn(session, SALARY)

    handle_command(session, "/example add")

    assert "An answer stopped by a guardrail cannot be kept" in capsys.readouterr().err
    assert kept(session) == []


def test_the_same_answer_is_kept_once(session, capsys):
    chat_turn(session, SALARY)
    handle_command(session, "/example add")
    handle_command(session, "/example add")
    assert "This answer is already kept as example 1" in capsys.readouterr().err
    assert len(kept(session)) == 1


def test_a_resumed_answer_can_be_kept_without_its_excerpts(session, capsys):
    session.agent.retriever = Retriever()
    load_memory(session, None)
    chat_turn(session, SALARY)
    handle_command(session, "/new")

    handle_command(session, "/resume")
    handle_command(session, "/example add")

    [example] = kept(session)
    assert (example.question, example.excerpts) == (SALARY, [])


def test_examples_lists_them_and_marks_those_shown(session, capsys):
    chat_turn(session, SALARY)
    handle_command(session, "/example add")
    session.examples.add(Example("Où est le siège d'Orion ?", "À Lyon [1].", created_at="2026-10-31T15:00:00+00:00"))
    chat_turn(session, BONUS)
    capsys.readouterr()

    handle_command(session, "/examples")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Answers kept as examples (* shown to the model for the last answer):"
    assert lines[1].startswith(f"  * 1. {SALARY} (kept on ")
    assert lines[2].startswith("    2. Où est le siège d'Orion ? (kept on 2026-10-31")


def test_examples_without_any(session, capsys):
    handle_command(session, "/examples")
    assert capsys.readouterr().out == "No examples yet: /example add after a good answer to keep it.\n"


def test_examples_says_when_the_feature_is_off(session, config, capsys):
    config.features["examples"] = False
    handle_command(session, "/examples")
    assert "The examples feature is off: /feature examples on" in capsys.readouterr().out


def test_remove_deletes_an_example(session, capsys):
    session.examples.add(Example(SALARY, ANSWER))
    handle_command(session, "/example remove 1")
    assert capsys.readouterr().out == f'Example 1 removed: "{SALARY}"\n'
    assert kept(session) == []


@pytest.mark.parametrize("argument", ["remove 3", "remove", "remove ²", "add 2", "keep", ""])
def test_example_rejects_wrong_arguments(session, capsys, argument):
    handle_command(session, f"/example {argument}")
    err = capsys.readouterr().err
    assert "No example 3: the library has 0" in err if argument == "remove 3" else "Usage: /example add" in err


def test_a_failed_embedding_is_a_warning(session, capsys):
    def broken(text: str) -> list[float]:
        raise LLMError("Cannot reach Ollama")

    session.examples.add(Example(SALARY, ANSWER))
    session.examples.embed = broken
    chat_turn(session, BONUS)
    assert "[warning] No examples were shown to the model: Cannot reach Ollama" in capsys.readouterr().err


def test_an_invalid_file_disables_the_examples(config, client, capsys):
    config.examples_path.write_text(json.dumps({"examples": [{"question": "Qui ?"}]}), encoding="utf-8")
    assert load_examples(config, client) is None
    assert "[warning] Examples disabled: Invalid examples file" in capsys.readouterr().err


def test_a_date_edited_by_hand_does_not_stop_the_cli(config, client, capsys):
    example = {"question": SALARY, "answer": ANSWER, "created_at": "yesterday"}
    config.examples_path.write_text(json.dumps({"examples": [example]}), encoding="utf-8")
    session = Session(config, Agent(config, client))
    session.examples = load_examples(config, client)
    handle_command(session, "/examples")
    out, err = capsys.readouterr()
    assert 'example 1 has an invalid date: "yesterday"' in err
    assert "The examples could not be loaded" in out


def test_a_missing_embedding_model_disables_the_examples(config, client, capsys):
    ExampleLibrary(config.examples_path, lambda text: [1.0]).add(Example(SALARY, ANSWER))

    def ensure_model(model: str) -> None:
        raise ModelNotFoundError(model, ["chat-model"])

    client.ensure_model = ensure_model
    assert load_examples(config, client) is None
    assert "[warning] Examples disabled: " in capsys.readouterr().err


def test_an_empty_library_does_not_need_the_embedding_model(config, client):
    def ensure_model(model: str) -> None:
        raise ModelNotFoundError(model, ["chat-model"])

    client.ensure_model = ensure_model
    assert load_examples(config, client).examples == []


def test_feature_examples_on_loads_them_again(session, config, capsys):
    session.examples = session.agent.examples = None
    handle_command(session, "/feature examples on")
    assert isinstance(session.examples, ExampleLibrary) and session.agent.examples is session.examples


def test_help_lists_the_example_commands():
    assert "/examples" in HELP and "/example add|remove N" in HELP

