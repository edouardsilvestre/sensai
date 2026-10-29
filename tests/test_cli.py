import pytest

from sensai.core.agent import Agent
from sensai.core.tools.registry import ToolRegistry
from sensai.interfaces.cli.main import Session, handle_command, run_status


@pytest.fixture
def session(config, client):
    return Session(config, Agent(config, client))


def lookup(query: str) -> str:
    """
    Look something up.

    Args:
      query: What to look up
    """
    return query


def test_features_lists_every_feature(session, capsys):
    handle_command(session, "/features")
    lines = [line.split(maxsplit=2) for line in capsys.readouterr().out.splitlines()]
    assert lines == [
        ["rag", "on", "answer from the indexed documents"],
        ["tools", "on", "let the model call the tools enabled in the configuration"],
        ["web_search", "off", "let the model search the web; the queries are sent to Ollama's web search"],
        ["code_exec", "off", "let the model run Python code in a Docker sandbox"],
        ["sessions", "on", "save every conversation so that it can be resumed"],
        ["profile", "on", "send your profile to the model at the start of every conversation"],
    ]


def test_feature_can_be_turned_off_and_on(session, capsys):
    handle_command(session, "/feature rag off")
    assert session.config.features["rag"] is False
    handle_command(session, "/feature RAG on")
    assert session.config.features["rag"] is True
    assert "No documents indexed yet" in capsys.readouterr().out


@pytest.mark.parametrize("argument", ["", "rag", "rag maybe", "unknown off"])
def test_invalid_feature_command(session, capsys, argument):
    handle_command(session, f"/feature {argument}")
    assert session.config.features["rag"] is True
    assert "Usage: /feature NAME on|off" in capsys.readouterr().err


def test_tools_feature_can_be_turned_off(session, capsys):
    session.agent.tools = ToolRegistry([lookup])
    handle_command(session, "/feature tools off")
    handle_command(session, "/tools")
    assert "The tools feature is off" in capsys.readouterr().out


def test_turning_tools_on_builds_them(config, client, capsys):
    config.features["tools"] = False
    config.features["web_search"] = True
    session = Session(config, Agent(config, client))
    handle_command(session, "/feature tools on")
    assert session.agent.tools.names() == ["web_search"]
    assert "Tools: web_search" in capsys.readouterr().out


def test_web_search_can_be_turned_on_and_off_during_a_chat(config, client, capsys):
    config.ollama_api_key = "secret"
    session = Session(config, Agent(config, client))
    handle_command(session, "/feature web_search on")
    assert session.agent.tools.names() == ["web_search"]
    handle_command(session, "/feature web_search off")
    assert not session.agent.tools
    handle_command(session, "/tools")
    out = capsys.readouterr().out
    assert "No tool is enabled, turn one on with /feature NAME on:" in out
    assert "  web_search: let the model search the web" in out
    assert "  code_exec: let the model run Python code" in out


def test_web_search_needs_the_api_key_to_be_turned_on(session, capsys):
    handle_command(session, "/feature web_search on")
    assert session.config.features["web_search"] is False and not session.agent.tools
    assert "web_search needs OLLAMA_API_KEY" in capsys.readouterr().err


def test_tools_lists_the_enabled_tools(session, capsys):
    session.agent.tools = ToolRegistry([lookup])
    handle_command(session, "/tools")
    assert "lookup: Look something up." in capsys.readouterr().out


def test_status_shows_the_features(config, capsys):
    config.features["rag"] = False
    run_status(config)
    assert "Features: rag off, tools on, web_search off, code_exec off, sessions on, profile on" in capsys.readouterr().out
