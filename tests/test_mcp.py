import asyncio
import json
import socket
import threading
import time

import pytest

from mcp_servers import calculator
from sensai.core.agent import Agent
from sensai.core.llm.ollama import ToolCall
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.mcp import MCPConfigError, MCPServer, _connect, call_tool, load_servers, load_tools
from sensai.core.tools.registry import ToolError, ToolRegistry, ToolResult
from sensai.interfaces.cli.main import Session, handle_command


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def calculator_url() -> str:
    # The real calculator server, over HTTP on a free port: the tests speak the protocol, without the internet.
    port = free_port()
    threading.Thread(
        target=calculator.server.run, args=("streamable-http",), kwargs={"port": port}, daemon=True,
    ).start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return f"http://127.0.0.1:{port}/mcp"
        time.sleep(0.05)
    pytest.fail("the calculator server did not start")


@pytest.fixture
def down_url() -> str:
    return f"http://127.0.0.1:{free_port()}/mcp"


def write(tmp_path, data) -> object:
    path = tmp_path / "mcp.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_without_file_there_is_no_server(tmp_path):
    assert load_servers(tmp_path / "mcp.json") == []


def test_servers_are_read_with_their_url(tmp_path):
    path = write(tmp_path, {"mcpServers": {
        "calculator": {"url": " http://127.0.0.1:8001/mcp "},
        "deepwiki": {"url": "https://mcp.deepwiki.com/mcp"},
    }})
    assert load_servers(path) == [
        MCPServer("calculator", "http://127.0.0.1:8001/mcp"),
        MCPServer("deepwiki", "https://mcp.deepwiki.com/mcp"),
    ]


def test_servers_are_read_with_their_headers(tmp_path):
    path = write(tmp_path, {"mcpServers": {
        "private": {"url": "https://mcp.example/mcp", "headers": {"Authorization": "Bearer key"}},
    }})
    [server] = load_servers(path)
    assert server.headers == {"Authorization": "Bearer key"}
    assert "Bearer key" not in repr(server)


@pytest.mark.parametrize(("data", "message"), [
    ("{not json", "Cannot read the MCP servers"),
    ({"servers": {}}, "expected an object with an 'mcpServers' object"),
    ({"mcpServers": {"local": {"command": "python", "args": ["server.py"]}}}, "'local' needs a 'url'"),
    ({"mcpServers": {"empty": {"url": " "}}}, "'empty' needs a 'url'"),
    ({"mcpServers": {"bad": {"url": "http://x/mcp", "headers": {"X-Retries": 3}}}}, "'headers' must map"),
])
def test_an_invalid_file_is_an_error(tmp_path, data, message):
    with pytest.raises(MCPConfigError, match=message):
        load_servers(write(tmp_path, data))


def test_the_tools_of_a_server_are_found(calculator_url):
    tools, errors = load_tools([MCPServer("calculator", calculator_url)], timeout=10)
    assert errors == []
    assert [tool.name for tool in tools] == ["add", "multiply"]
    assert tools[0].parameters["required"] == ["a", "b"]
    assert tools[0].description.strip().startswith("Add two numbers.")
    assert tools[0].approval_reason == "the arguments are sent to the MCP server 'calculator'"


def test_a_server_with_headers_is_reached_too(calculator_url):
    server = MCPServer("calculator", calculator_url, {"Authorization": "Bearer key"})
    assert call_tool(server, 10, "add", {"a": 2, "b": 3}) == ToolResult("5.0")


def test_a_tool_is_run_by_its_server(calculator_url):
    registry = ToolRegistry(load_tools([MCPServer("calculator", calculator_url)], timeout=10)[0])
    assert registry.execute(ToolCall("multiply", {"a": 6, "b": 7})) == ToolResult("42.0")


def test_the_server_checks_the_arguments(calculator_url):
    result = call_tool(MCPServer("calculator", calculator_url), 10, "add", {"a": "x"})
    assert result.error and result.content.startswith("Error: ") and "validation error" in result.content


def test_a_name_already_used_is_left_out(calculator_url):
    tools, errors = load_tools([MCPServer("calculator", calculator_url)], timeout=10, taken=["add"])
    assert [tool.name for tool in tools] == ["multiply"]
    assert errors == ["MCP server 'calculator': add left out, another tool already has this name"]


def test_an_unreachable_server_is_skipped(calculator_url, down_url):
    servers = [MCPServer("down", down_url), MCPServer("calculator", calculator_url)]
    tools, errors = load_tools(servers, timeout=10)
    assert [tool.name for tool in tools] == ["add", "multiply"]
    assert len(errors) == 1
    assert errors[0].startswith(f"MCP server 'down' failed ({down_url}): ")
    assert errors[0].endswith(": its tools are disabled")


def test_a_server_that_stops_answers_with_an_error(down_url):
    registry = ToolRegistry(load_tools([], timeout=10)[0])
    assert not registry
    with pytest.raises(ToolError, match="MCP server 'down' failed"):
        call_tool(MCPServer("down", down_url), 10, "add", {"a": 1, "b": 2})


def test_a_slow_server_times_out(calculator_url):
    async def slow(client):
        await asyncio.sleep(5)

    with pytest.raises(ToolError, match="did not answer within 1 seconds"):
        _connect(MCPServer("calculator", calculator_url), 1, slow)


def servers_file(config, **urls) -> None:
    config.mcp_path.write_text(json.dumps({"mcpServers": {name: {"url": url} for name, url in urls.items()}}))


def test_the_registry_adds_the_mcp_tools_after_the_others(config, client, calculator_url):
    servers_file(config, calculator=calculator_url)
    registry = build_registry(config, client)
    assert registry.names()[-2:] == ["add", "multiply"] and "memory_search" in registry.names()
    assert registry.warnings == []
    assert registry.execute(ToolCall("add", {"a": 1, "b": 2})) == ToolResult("3.0")


def test_a_server_that_is_down_only_disables_its_tools(config, client, down_url):
    servers_file(config, down=down_url)
    registry = build_registry(config, client)
    assert registry.names()[0] == "memory_search"
    assert len(registry.warnings) == 1 and registry.warnings[0].startswith("MCP server 'down' failed")


def test_no_server_is_reached_when_the_mcp_feature_is_off(config, client, down_url):
    servers_file(config, down=down_url)
    config.features["mcp"] = False
    assert build_registry(config, client).warnings == []


def test_an_invalid_mcp_file_stops_the_registry(config, client):
    config.mcp_path.write_text("{not json")
    with pytest.raises(MCPConfigError):
        build_registry(config, client)


def test_turning_mcp_on_in_the_chat_shows_the_servers_that_are_down(config, client, capsys, calculator_url, down_url):
    servers_file(config, calculator=calculator_url, down=down_url)
    config.features["mcp"] = False
    session = Session(config, Agent(config, client))
    handle_command(session, "/feature mcp on")
    captured = capsys.readouterr()
    assert "add" in session.agent.tools.names()
    assert "[warning] MCP server 'down' failed" in captured.err
    assert "add, multiply" in captured.out


def test_an_invalid_mcp_file_is_reported_in_the_chat(config, client, capsys):
    config.mcp_path.write_text("{not json")
    config.features["mcp"] = False
    session = Session(config, Agent(config, client))
    handle_command(session, "/feature mcp on")
    assert "[error] Cannot read the MCP servers" in capsys.readouterr().err
