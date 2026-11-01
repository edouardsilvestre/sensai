import json

import pytest

from sensai.core.agent import Agent
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.persona import Persona, PersonaError, load_persona, load_personas
from sensai.interfaces.cli.commands.personas import persona_command
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import load_persona as load_startup_persona

INVESTIGATOR = Persona("investigator", "Investigator", "a researcher.", "neutral.", "find leads.")
EDITOR = Persona("editor", "Editor", "an editor.", "clear.", "write drafts.")


def write_persona(directory, persona_id, **fields):
    data = {"name": "Legal", "role": "a lawyer.", "tone": "cautious.", "scope": "flag risks.", **fields}
    path = directory / f"{persona_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def chat(agent, message):
    return list(agent.run(message))


def system_prompts(messages):
    return [message["content"] for message in messages if message["role"] == "system"]


def test_load_persona_uses_the_file_name_as_id(tmp_path):
    persona = load_persona(write_persona(tmp_path, "legal", instructions="Cite the law."))
    assert persona == Persona("legal", "Legal", "a lawyer.", "cautious.", "flag risks.", "Cite the law.")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", json.dumps({"name": "Legal", "role": "a lawyer."})])
def test_invalid_persona_file_names_the_file(tmp_path, content):
    path = tmp_path / "legal.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PersonaError, match="legal.json"):
        load_persona(path)


def test_missing_fields_are_listed(tmp_path):
    with pytest.raises(PersonaError, match="tone, scope"):
        load_persona(write_persona(tmp_path, "legal", tone="", scope=" "))


def test_missing_directory_means_no_persona(tmp_path):
    assert load_personas(tmp_path / "nope") == {}


def test_shipped_personas_are_valid():
    assert set(load_personas(DEFAULT_PERSONAS_DIR)) == {"investigator", "fact_checker", "editor"}


def test_the_scope_is_not_a_task_to_start():
    prompt = Persona("p", "Ana", "a role.", "calm.", "find leads.").system_prompt()
    assert "Answer the user's message and nothing else." in prompt
    assert "When the user asks for it, you can help with this work: find leads." in prompt


def test_system_prompt_contains_every_field():
    prompt = Persona("p", "Ana", "a role.", "calm.", "only facts.", "Cite.").system_prompt()
    for text in ("Ana", "a role.", "calm.", "only facts.", "Cite."):
        assert text in prompt


def test_persona_replaces_the_configured_system_prompt(config, client):
    config.system_prompt = "plain prompt"
    agent = Agent(config, client, persona=INVESTIGATOR)
    assert system_prompts(agent.messages) == [INVESTIGATOR.system_prompt()]


def test_without_persona_the_configured_system_prompt_is_used(config, client):
    config.system_prompt = "plain prompt"
    assert system_prompts(Agent(config, client).messages) == ["plain prompt"]


def test_switching_persona_keeps_the_history(config, client):
    agent = Agent(config, client, persona=INVESTIGATOR)
    chat(agent, "Orion paid 2M to Delta.")

    agent.set_persona(EDITOR)
    chat(agent, "Write a headline.")

    sent, _ = client.chats[-1]
    assert system_prompts(sent) == [EDITOR.system_prompt()]
    assert sent[1:3] == [
        {"role": "user", "content": "Orion paid 2M to Delta."},
        {"role": "assistant", "content": "Hello there "},
    ]


def test_switching_persona_keeps_the_summary_of_older_turns(config, client):
    agent = Agent(config, client, persona=INVESTIGATOR)
    chat(agent, "Orion paid 2M to Delta.")
    agent.memory.compact()

    agent.set_persona(EDITOR)

    [prompt] = system_prompts(agent.messages)
    assert prompt.startswith(EDITOR.system_prompt())
    assert client.summary in prompt


def test_unknown_startup_persona_lists_the_available_ones(config, tmp_path):
    write_persona(tmp_path, "legal")
    config.personas_dir, config.persona = tmp_path, "nope"
    with pytest.raises(PersonaError, match="Available: legal"):
        load_startup_persona(config)


def test_empty_startup_persona_means_none(config, tmp_path):
    write_persona(tmp_path, "legal")
    config.personas_dir, config.persona = tmp_path, ""
    assert load_startup_persona(config) == ({"legal": load_persona(tmp_path / "legal.json")}, None)


def test_persona_command_switches_and_rejects_unknown_ids(config, client, capsys):
    session = Session(config, Agent(config, client, persona=INVESTIGATOR), personas={"editor": EDITOR})

    persona_command(session, "nope")
    assert session.agent.persona == INVESTIGATOR
    assert "Unknown persona 'nope'. Available: editor" in capsys.readouterr().err

    persona_command(session, "editor")
    assert session.agent.persona == EDITOR
    assert "history is kept" in capsys.readouterr().out
