import json

import pytest

from sensai.core.llm.ollama import ToolCall
from sensai.core.reasoning import ANSWER, ReasoningError, Step, instructions, parse_step, step_schema
from sensai.core.tools.registry import SchemaTool, ToolRegistry

ACTIONS = ["search_documents", "web_search"]


def search_documents(query: str) -> str:
    """
    Search the journalist's indexed documents.

    Args:
      query: What to look for

    Returns:
      The excerpts found
    """
    return query


def lookup_company(arguments: dict) -> str:
    return arguments["siren"]


COMPANY = SchemaTool(
    "lookup_company",
    "Look up a French company in the SIRENE register.\n\nThe SIREN has 9 digits.",
    {"type": "object", "properties": {"siren": {"type": "string"}}, "required": ["siren"]},
    lookup_company,
)


def step(**fields) -> str:
    return json.dumps({"thought": "I need the manager of Orion.", "action": "search_documents",
                       "arguments": {"query": "Orion manager"}} | fields)


def test_the_schema_only_allows_the_tools_and_the_answer():
    forms = step_schema(ToolRegistry([search_documents, COMPANY]))["anyOf"]
    assert [form["properties"]["action"] for form in forms] == [
        {"const": "search_documents"}, {"const": "lookup_company"}, {"const": ANSWER}
    ]
    assert all(form["required"] == ["thought", "action", "arguments"] for form in forms)


def test_each_tool_only_takes_its_own_arguments():
    search, company, answer = (form["properties"]["arguments"] for form in step_schema(
        ToolRegistry([search_documents, COMPANY]))["anyOf"])
    assert search == {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
        "additionalProperties": False,
    }
    assert company == {**COMPANY.parameters, "additionalProperties": False}
    assert answer == {"type": "object", "properties": {}, "additionalProperties": False}


def test_a_step_is_read_from_the_reply():
    parsed = parse_step(step(), ACTIONS)
    assert parsed == Step("I need the manager of Orion.", "search_documents", {"query": "Orion manager"})
    assert not parsed.final
    assert parsed.call() == ToolCall("search_documents", {"query": "Orion manager"})


def test_the_answer_ends_the_reasoning():
    parsed = parse_step(step(thought="  Marc Vidal manages Orion.  ", action=ANSWER, arguments={}), ACTIONS)
    assert parsed == Step("Marc Vidal manages Orion.", ANSWER, {})
    assert parsed.final


def test_missing_arguments_are_empty():
    assert parse_step('{"thought": "I know enough.", "action": "answer"}', ACTIONS).arguments == {}


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ('{"thought": "I need the manager', "not valid JSON"),
        ('["search_documents"]', "not a JSON object"),
        (step(thought="  "), "no thought"),
        (step(action="delete_files"), "Unknown action 'delete_files'. Available: search_documents, web_search, answer"),
        (step(arguments="Orion manager"), "arguments of the step are not a JSON object"),
    ],
)
def test_an_invalid_step_is_refused(reply, message):
    with pytest.raises(ReasoningError, match=message):
        parse_step(reply, ACTIONS)


def test_the_instructions_list_the_tools_and_their_arguments():
    text = instructions(ToolRegistry([search_documents, COMPANY]))
    assert "- search_documents(query): Search the journalist's indexed documents." in text
    assert "- lookup_company(siren): Look up a French company in the SIRENE register." in text
    assert '{"query": "..."}' in text


def test_without_tools_the_instructions_ask_for_the_answer():
    assert 'Tools:\nnone, pick "answer".' in instructions(ToolRegistry())
