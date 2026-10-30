import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from sensai.core.tools.registry import SchemaTool, ToolError, ToolResult


class MCPConfigError(Exception):
    pass


@dataclass
class MCPServer:
    """A server of the MCP file, reached by its URL (Streamable HTTP transport)."""

    name: str
    url: str
    # Sent with every request, like the token of a server that asks for one. Kept out of the repr: it can be a secret.
    headers: dict[str, str] = field(default_factory=dict, repr=False)


def load_servers(path: Path) -> list[MCPServer]:
    """The servers of the file, none when it does not exist. An invalid file is an error: no guess."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise MCPConfigError(f"Cannot read the MCP servers {path}: {e}") from None
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise MCPConfigError(f"Invalid MCP servers {path}: expected an object with an 'mcpServers' object")
    return [_server(path, name, entry) for name, entry in servers.items()]


def _server(path: Path, name: str, entry: Any) -> MCPServer:
    where = f"Invalid MCP servers {path}: '{name}'"
    url = entry.get("url") if isinstance(entry, dict) else None
    if not isinstance(url, str) or not url.strip():
        raise MCPConfigError(f"{where} needs a 'url': only servers reached by their URL are supported")
    headers = entry.get("headers", {})
    if not isinstance(headers, dict) or not all(isinstance(value, str) for value in headers.values()):
        raise MCPConfigError(f"{where}: 'headers' must map each header name to a text")
    return MCPServer(name, url.strip(), headers)


def _cause(error: BaseException) -> str:
    # The SDK raises the failures of its task groups wrapped in groups: the first one inside says what happened.
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return str(error) or type(error).__name__


def _connect(server: MCPServer, timeout: int, work: Callable[[Client], Awaitable[Any]]) -> Any:
    """
    Connects to the server, runs `work` with the client and disconnects.
    One connection per call: nothing stays open between two calls, and a restarted server is simply reached again.
    """

    async def run() -> Any:
        if not server.headers:
            async with Client(server.url) as client:
                return await work(client)
        # The SDK only takes headers through an HTTP client of its own library.
        async with httpx2.AsyncClient(headers=server.headers, timeout=timeout) as http:
            async with Client(streamable_http_client(server.url, http_client=http)) as client:
                return await work(client)

    try:
        return asyncio.run(asyncio.wait_for(run(), timeout))
    except TimeoutError:
        raise ToolError(f"MCP server '{server.name}' did not answer within {timeout} seconds") from None
    except Exception as e:
        raise ToolError(f"MCP server '{server.name}' failed ({server.url}): {_cause(e)}") from None


def _result(result: Any) -> ToolResult:
    # A tool answers with blocks: the model reads the text, the other kinds (images, files) are only named.
    parts = [block.text if block.type == "text" else f"[{block.type} not shown]" for block in result.content]
    text = "\n".join(parts)
    return ToolResult(f"Error: {text}" if result.is_error else text, error=result.is_error)


def call_tool(server: MCPServer, timeout: int, name: str, arguments: dict) -> ToolResult:
    async def work(client: Client) -> Any:
        return await client.call_tool(name, arguments)

    return _result(_connect(server, timeout, work))


def load_tools(
    servers: list[MCPServer], timeout: int, taken: Iterable[str] = ()
) -> tuple[list[SchemaTool], list[str]]:
    """
    The tools of every server that answers, and why each other server or tool is left out.
    `taken` holds the names of the other tools: a tool whose name is already used is left out, never replaced.
    """

    async def work(client: Client) -> Any:
        return (await client.list_tools()).tools

    tools: list[SchemaTool] = []
    errors: list[str] = []
    names = set(taken)
    for server in servers:
        try:
            found = _connect(server, timeout, work)
        except ToolError as e:
            errors.append(f"{e}: its tools are disabled")
            continue
        for tool in found:
            if tool.name in names:
                errors.append(f"MCP server '{server.name}': {tool.name} left out, another tool already has this name")
                continue
            names.add(tool.name)
            tools.append(SchemaTool(
                tool.name, tool.description or "", tool.input_schema,
                partial(call_tool, server, timeout, tool.name),
                approval_reason=f"the arguments are sent to the MCP server '{server.name}'",
            ))
    return tools, errors
