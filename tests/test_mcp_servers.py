import asyncio
from urllib.error import URLError

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from mcp_servers import calculator, sirene

EPITECH = {
    "nom_complet": "EPITECH",
    "siren": "417617776",
    "etat_administratif": "A",
    "siege": {"siret": "41761777600154", "adresse": "20 RUE JOUBERT 75009 PARIS"},
}


def call(server, name: str, arguments: dict):
    # The SDK client connects to the server in memory, the way a real client talks to it over stdio.
    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def tool_names(server) -> list[str]:
    async def run():
        async with Client(server) as client:
            return [tool.name for tool in (await client.list_tools()).tools]

    return asyncio.run(run())


def test_calculator_lists_its_tools():
    assert tool_names(calculator.server) == ["add", "multiply"]


def test_calculator_computes():
    assert call(calculator.server, "add", {"a": 2, "b": 3.5}).content[0].text == "5.5"
    assert call(calculator.server, "multiply", {"a": 4, "b": 2.5}).content[0].text == "10.0"


def test_calculator_rejects_wrong_arguments():
    assert call(calculator.server, "add", {"a": "x"}).is_error


def test_find_siren_describes_the_matches(monkeypatch):
    monkeypatch.setattr(sirene, "search", lambda name: [EPITECH, {**EPITECH, "etat_administratif": "C"}])
    result = call(sirene.server, "find_siren", {"name": " Epitech "})
    assert not result.is_error
    assert result.content[0].text == (
        "EPITECH: SIREN 417617776, head office SIRET 41761777600154, 20 RUE JOUBERT 75009 PARIS (active)\n"
        "EPITECH: SIREN 417617776, head office SIRET 41761777600154, 20 RUE JOUBERT 75009 PARIS (closed)"
    )


def test_find_siren_keeps_the_best_matches(monkeypatch):
    monkeypatch.setattr(sirene, "search", lambda name: [EPITECH] * 5)
    assert len(sirene.find_siren("Epitech").splitlines()) == sirene.MAX_RESULTS


def test_find_siren_without_match(monkeypatch):
    monkeypatch.setattr(sirene, "search", lambda name: [])
    assert sirene.find_siren("Nobody") == "No company found for 'Nobody'"


def test_find_siren_rejects_an_empty_name():
    result = call(sirene.server, "find_siren", {"name": "  "})
    assert result.is_error and "the name is empty" in result.content[0].text


def test_search_reports_an_unreachable_api(monkeypatch):
    def fail(url, timeout):
        raise URLError("no network")

    monkeypatch.setattr(sirene, "urlopen", fail)
    with pytest.raises(ToolError, match="did not answer"):
        sirene.search("Epitech")
