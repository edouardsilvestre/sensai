import json

import pytest

from sensai.core.agent import Agent
from sensai.core.llm.ollama import LLMError
from sensai.core.memory.sessions import SessionStore
from sensai.core.prompts import PromptStore
from sensai.interfaces.cli.answer import chat_turn
from sensai.interfaces.cli.commands.dispatch import HELP, handle_command
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import load_prompts


def start(config, client) -> Session:
    prompts = load_prompts(config)
    agent = Agent(config, client, prompts=prompts)
    return Session(config, agent, prompts=prompts)


@pytest.fixture
def session(config, client):
    config.features["document"] = False
    config.system_prompt = "Initial system prompt."
    start(config, client)
    config.system_prompt = "Updated system prompt."
    return start(config, client)


def system_prompt(session) -> str:
    return session.agent.messages[0]["content"]


def test_startup_keeps_system_prompt_version(config, client, capsys):
    config.system_prompt = "Initial system prompt."
    session = start(config, client)
    assert len(session.prompts.history().versions) == 1
    assert session.agent.prompt_version == 1
    assert "changed" not in capsys.readouterr().out


def test_an_edited_system_prompt_is_a_new_version(config, client, capsys):
    config.system_prompt = "Initial system prompt."
    start(config, client)
    config.system_prompt = "Updated system prompt."
    session = start(config, client)
    assert "System prompt changed: kept as version 2" in capsys.readouterr().out
    assert session.agent.prompt_version == 2 and system_prompt(session) == "Updated system prompt."


def test_prompts_lists_the_versions_with_the_active_one(session, capsys):
    handle_command(session, "/prompts")
    out = capsys.readouterr().out
    assert "Versions of the system prompt" in out
    assert "  * 2. " in out and "    1. " in out


def test_use_rolls_back_the_system_prompt(session, capsys):
    handle_command(session, "/prompt use 1")
    assert "Sensai uses version 1 of its system prompt from now on." in capsys.readouterr().out
    assert system_prompt(session) == "Initial system prompt." and session.agent.prompt_version == 1
    handle_command(session, "/prompts")
    assert "  * 1. " in capsys.readouterr().out


def test_rollback_is_kept_at_the_next_start(session, config, client):
    handle_command(session, "/prompt use 1")
    restarted = start(config, client)
    assert system_prompt(restarted) == "Initial system prompt."


def test_show_and_diff(session, capsys):
    handle_command(session, "/prompt show 1")
    out = capsys.readouterr().out
    assert "Version 1 of the system prompt" in out and "Initial system prompt." in out
    handle_command(session, "/prompt diff 1 2")
    out = capsys.readouterr().out
    assert "-Initial system prompt." in out and "+Updated system prompt." in out
    handle_command(session, "/prompt diff 2 2")
    assert "Versions 2 and 2 are identical." in capsys.readouterr().out


@pytest.mark.parametrize("command", ["/prompt", "/prompt use", "/prompt use x", "/prompt diff 1", "/prompt run 1"])
def test_invalid_prompt_command(session, capsys, command):
    handle_command(session, command)
    assert "Usage: /prompt show|use N, or /prompt diff|compare N M" in capsys.readouterr().err


def test_unknown_version(session, capsys):
    handle_command(session, "/prompt use 9")
    assert "No version 9 of system prompt: it has 2" in capsys.readouterr().err


def test_feature_off_uses_the_config_prompt(session, capsys, config):
    handle_command(session, "/prompt use 1")
    handle_command(session, "/feature prompt_versions off")
    assert system_prompt(session) == config.system_prompt and session.agent.prompt_version is None
    handle_command(session, "/prompts")
    assert "prompt_versions feature is off" in capsys.readouterr().out
    handle_command(session, "/feature prompt_versions on")
    assert system_prompt(session) == "Initial system prompt." and session.agent.prompt_version == 1


def test_invalid_versions_file_is_a_warning(config, client, capsys):
    config.system_prompt = "Initial prompt."
    config.prompts_dir.mkdir()
    (config.prompts_dir / "system.json").write_text("{not json", encoding="utf-8")
    session = start(config, client)
    assert "Prompt versions disabled" in capsys.readouterr().err
    assert session.prompts is None and system_prompt(session) == "Initial prompt."


def test_session_records_the_prompt_version(session):
    session.store = SessionStore(session.config.sessions_dir)
    chat_turn(session, "Hello")
    data = json.loads(next(session.config.sessions_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert (data["persona"], data["prompt_version"]) == (None, 2)


def write_cases(config):
    cases = [
        {"question": "Who signed?", "must_contain": ["Vidal"]},
        {"question": "Rumours?", "must_not_contain": ["rumour"]},
    ]
    config.prompt_cases_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")


def test_compare_scores_both_versions_and_keeps_the_result(session, client, capsys):
    write_cases(session.config)
    client.chat = lambda model, messages, options=None, think=None, usage=None: (
        "Marc Vidal signed." if "Updated" in messages[0]["content"] else "A rumour says Vidal signed."
    )
    handle_command(session, "/prompt compare 1 2")
    out = capsys.readouterr().out
    assert "Comparing versions 1 and 2 on the test cases with chat-model..." in out
    rows = [line.split()[:2] for line in out.splitlines() if line[:1].isdigit()]
    assert rows == [["1", "1/2"], ["2", "2/2"]]
    assert 'version 1, case 2 (Rumours?): contains "rumour"' in out

    comparisons = PromptStore(session.config.prompts_dir).history().comparisons
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
    assert PromptStore(session.config.prompts_dir).history().comparisons == []


def test_help_lists_the_prompt_commands():
    assert "/prompts" in HELP and "/prompt compare N M" in HELP
    assert "/prompts [PERSONA]" not in HELP
