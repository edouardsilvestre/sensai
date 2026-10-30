from pathlib import Path

import pytest

from sensai.core.agent import Agent
from sensai.core.approval import Approval
from sensai.core.events import ApprovalEvent, ToolResultEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import ToolCall
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.files import FileTools
from sensai.core.tools.permissions import Permissions
from sensai.core.tools.registry import ToolRegistry
from sensai.interfaces.cli.main import Session, handle_command

WRITE = ToolCall("write_file", {"path": "drafts/article.md", "content": "Claire Benoit a refusé le virement."})


@pytest.fixture
def agent(config, client, tmp_path) -> Agent:
    permissions = Permissions(read=["leak"], write=["drafts"], base=tmp_path)
    source = {"name": "source", "terms": ["Claire Benoit"], "on": ["output", "tools"], "replacement": "[SOURCE]"}
    agent = Agent(config, client, tools=ToolRegistry(FileTools(permissions).tools()))
    agent.guardrails = Guardrails(parse_rules([source]))
    client.tool_rounds = [[WRITE]]
    return agent


def test_the_file_tools_are_built_only_with_the_files_feature(config, client):
    assert "read_file" not in build_registry(config, client).names()
    config.features["files"] = True
    assert build_registry(config, client).names()[-3:] == ["list_files", "read_file", "write_file"]


def test_a_refused_write_writes_nothing(agent, tmp_path):
    agent.approver = lambda call, reason: Approval(False, "pas encore")
    events = list(agent.run("Écris le brouillon"))
    assert [event.approval.approved for event in events if isinstance(event, ApprovalEvent)] == [False]
    assert not (tmp_path / "drafts" / "article.md").exists()


def test_an_approved_write_is_redacted_by_the_guardrails(agent, tmp_path):
    asked = []
    agent.approver = lambda call, reason: asked.append((call, reason)) or Approval(True)
    events = list(agent.run("Écris le brouillon"))
    assert asked[0][1] == "it writes a file on this machine"
    assert asked[0][0].arguments["content"] == "[SOURCE] a refusé le virement."
    assert (tmp_path / "drafts" / "article.md").read_text(encoding="utf-8") == "[SOURCE] a refusé le virement."
    assert not next(event.result for event in events if isinstance(event, ToolResultEvent)).error


def test_files_shows_the_permissions(config, client, capsys):
    config.files_read, config.files_write, config.files_deny = [Path(".")], [Path("missing-drafts")], ["sources"]
    handle_command(Session(config, Agent(config, client)), "/files")
    out = capsys.readouterr().out
    assert "The files feature is off: /feature files on" in out
    assert "Read (SENSAI_FILES_READ): .\n" in out
    assert "Read and write (SENSAI_FILES_WRITE): missing-drafts (missing)" in out
    assert "Always denied: .env, .env.*, .sensai, .git, sources" in out
    assert "Sensai asks before every file written" in out
