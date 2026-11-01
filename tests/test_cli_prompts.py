import json

import pytest

from sensai.core.agent import Agent
from sensai.core.llm.ollama import LLMError
from sensai.core.memory.sessions import SessionStore
from sensai.core.persona import Persona
from sensai.core.prompts import PromptStore
from sensai.interfaces.cli.main import HELP, Session, chat_turn, handle_command, load_prompts

EDITOR = Persona("editor", "Editor", "an editor.", "clear.", "write drafts.")
EDITED = Persona("editor", "Editor", "an editor.", "short.", "write drafts.")
INVESTIGATOR = Persona("investigator", "Investigator", "a researcher.", "neutral.", "find leads.")


def start(config, client, *personas: Persona, active: Persona | None = EDITOR) -> Session:
    by_id = {persona.id: persona for persona in personas or (EDITOR, INVESTIGATOR)}
    prompts = load_prompts(config, by_id)
    agent = Agent(config, client, persona=active, prompts=prompts)
    return Session(config, agent, personas=by_id, prompts=prompts)


@pytest.fixture
def session(config, client):
    config.features["document"] = False
    start(config, client)
    # The persona file was edited between two starts.
    return start(config, client, EDITED, INVESTIGATOR, active=EDITED)


def system_prompt(session) -> str:
    return session.agent.messages[0]["content"]


def test_startup_keeps_a_version_per_persona(config, client, capsys):
    session = start(config, client)
    assert [len(session.prompts.history(name).versions) for name in ("editor", "investigator")] == [1, 1]
    assert session.agent.prompt_version == 1
    assert "changed" not in capsys.readouterr().out


def test_an_edited_persona_file_is_a_new_version(config, client, capsys):
    start(config, client)
    session = start(config, client, EDITED, INVESTIGATOR, active=EDITED)
    assert "Prompt of editor changed: kept as version 2" in capsys.readouterr().out
    assert session.agent.prompt_version == 2 and system_prompt(session) == EDITED.system_prompt()


def test_prompts_lists_the_versions_with_the_active_one(session, capsys):
    handle_command(session, "/prompts")
    out = capsys.readouterr().out
    assert "Versions of the editor prompt" in out
    assert "  * 2. " in out and "    1. " in out


def test_use_rolls_back_the_active_persona(session, capsys):
    handle_command(session, "/prompt use 1")
    assert "editor uses version 1 of its prompt from now on." in capsys.readouterr().out
    assert system_prompt(session) == EDITOR.system_prompt() and session.agent.prompt_version == 1
    handle_command(session, "/prompts editor")
    assert "  * 1. " in capsys.readouterr().out


def test_rollback_is_kept_at_the_next_start(session, config, client):
    handle_command(session, "/prompt use 1")
    restarted = start(config, client, EDITED, INVESTIGATOR, active=EDITED)
    assert system_prompt(restarted) == EDITOR.system_prompt()


def test_show_and_diff(session, capsys):
    handle_command(session, "/prompt show 1")
    out = capsys.readouterr().out
    assert "Version 1 of the editor prompt" in out and "Tone: clear." in out
    handle_command(session, "/prompt diff 1 2")
    out = capsys.readouterr().out
    assert "-Tone: clear." in out and "+Tone: short." in out
    handle_command(session, "/prompt diff 2 2")
    assert "Versions 2 and 2 are identical." in capsys.readouterr().out


def test_another_persona_can_be_named(session, capsys):
    handle_command(session, "/prompt show 1 investigator")
    assert "Tone: neutral." in capsys.readouterr().out


@pytest.mark.parametrize("command", ["/prompt", "/prompt use", "/prompt use x", "/prompt diff 1", "/prompt run 1"])
def test_invalid_prompt_command(session, capsys, command):
    handle_command(session, command)
    assert "Usage: /prompt show|use N [PERSONA]" in capsys.readouterr().err


def test_unknown_version_and_persona(session, capsys):
    handle_command(session, "/prompt use 9")
    assert "No version 9 of editor: it has 2" in capsys.readouterr().err
    handle_command(session, "/prompts legal")
    assert "Unknown persona 'legal'" in capsys.readouterr().err


def test_without_active_persona_one_must_be_named(config, client, capsys):
    session = start(config, client, active=None)
    handle_command(session, "/prompts")
    assert "No active persona: name one" in capsys.readouterr().err


def test_feature_off_uses_the_persona_file(session, capsys):
    handle_command(session, "/prompt use 1")
    handle_command(session, "/feature prompt_versions off")
    assert system_prompt(session) == EDITED.system_prompt() and session.agent.prompt_version is None
    handle_command(session, "/prompts")
    assert "prompt_versions feature is off" in capsys.readouterr().out
    handle_command(session, "/feature prompt_versions on")
    assert system_prompt(session) == EDITOR.system_prompt()


def test_invalid_versions_file_is_a_warning(config, client, capsys):
    config.prompts_dir.mkdir()
    (config.prompts_dir / "editor.json").write_text("{not json", encoding="utf-8")
    session = start(config, client)
    assert "Prompt versions disabled" in capsys.readouterr().err
    assert session.prompts is None and system_prompt(session) == EDITOR.system_prompt()


def test_session_records_the_prompt_version(session):
    session.store = SessionStore(session.config.sessions_dir)
    chat_turn(session, "Write the headline.")
    data = json.loads(next(session.config.sessions_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert (data["persona"], data["prompt_version"]) == ("editor", 2)


def write_cases(config):
    cases = [{"question": "Who signed?", "must_contain": ["Vidal"]}, {"question": "Rumours?", "must_not_contain": ["rumour"]}]
    config.prompt_cases_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")


def test_compare_scores_both_versions_and_keeps_the_result(session, client, capsys):
    write_cases(session.config)
    # Version 1 answers with a rumour, version 2 sticks to the facts.
    client.chat = lambda model, messages, options=None, think=None, usage=None: (
        "Marc Vidal signed." if "short." in messages[0]["content"] else "A rumour says Vidal signed."
    )
    handle_command(session, "/prompt compare 1 2")
    out = capsys.readouterr().out
    assert "Comparing versions 1 and 2 on the test cases with chat-model..." in out
    rows = [line.split()[:2] for line in out.splitlines() if line[:1].isdigit()]
    assert rows == [["1", "1/2"], ["2", "2/2"]]
    assert 'version 1, case 2 (Rumours?): contains "rumour"' in out

    comparisons = PromptStore(session.config.prompts_dir).history("editor").comparisons
    assert [(s["version"], s["passed"]) for s in comparisons[-1]["scores"]] == [(1, 1), (2, 2)]
    handle_command(session, "/prompts")
    assert "Last comparison (chat-model): version 1 1/2, version 2 2/2." in capsys.readouterr().out


def test_compare_needs_test_cases(session, capsys):
    handle_command(session, "/prompt compare 1 2")
    assert "No test cases at" in capsys.readouterr().err


def test_compare_reports_a_model_error(session, client, capsys):
    write_cases(session.config)

    def fail(*args, **kwargs):
        raise LLMError("Ollama is down")

    client.chat = fail
    handle_command(session, "/prompt compare 1 2")
    assert "Ollama is down" in capsys.readouterr().err
    assert PromptStore(session.config.prompts_dir).history("editor").comparisons == []


def test_help_lists_the_prompt_commands():
    assert "/prompts [PERSONA]" in HELP and "/prompt compare N M [PERSONA]" in HELP
