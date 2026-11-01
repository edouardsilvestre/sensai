import json

import pytest

from conftest import FakeClient
from sensai.core.agent import Agent
from sensai.core.rag.models import SearchResult
from sensai.core.structured import ANSWER_SCHEMA, StructuredError, cited_sources, parse_answer, schema_errors
from sensai.interfaces.cli.main import Session, handle_command

ANSWER = {
    "answer": "Orion paid Delta 2M on 4 April.",
    "claims": [{"text": "Orion paid Delta 2M", "sources": [2, 1]}, {"text": "on 4 April", "sources": [2]}],
    "entities": [{"name": "Orion", "type": "company"}, {"name": "Delta", "type": "company"}],
    "events": [{"date": "2024-04-04", "description": "Orion pays Delta 2M"}],
}
RESULT = SearchResult(
    chunk_id=1, text="Orion paid Delta.", location="p. 2", source="leak/mail.eml", description="", score=0.9,
)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
        return self.results


def test_an_answer_that_follows_the_schema_is_read():
    assert parse_answer(json.dumps(ANSWER)) == ANSWER
    assert cited_sources(ANSWER) == [2, 1]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"answer": None}, "$.answer is not of type string"),
        ({"claims": [{"text": "Orion paid Delta"}]}, "$.claims[0].sources is missing"),
        ({"claims": [{"text": "Orion paid Delta", "sources": ["1"]}]}, "$.claims[0].sources[0] is not of type integer"),
        ({"claims": [{"text": "Orion paid Delta", "sources": [True]}]}, "$.claims[0].sources[0] is not of type integer"),
        ({"entities": [{"name": "Orion", "type": "bank"}]}, "$.entities[0].type is not one of person, company"),
        ({"events": {}}, "$.events is not of type array"),
    ],
)
def test_an_answer_that_does_not_follow_the_schema_is_refused(change, error):
    with pytest.raises(StructuredError, match=error.replace("$", r"\$").replace("[", r"\[").replace("]", r"\]")):
        parse_answer(json.dumps(ANSWER | change))


def test_a_missing_field_is_named():
    assert schema_errors({"answer": "Hi"}, ANSWER_SCHEMA) == [
        "$.claims is missing", "$.entities is missing", "$.events is missing",
    ]


@pytest.mark.parametrize("reply", ["Orion paid Delta.", '{"answer": "Orion paid', "[]"])
def test_an_answer_that_is_not_the_object_is_refused(reply):
    with pytest.raises(StructuredError):
        parse_answer(reply)


def test_json_command_prints_the_checked_answer(config, capsys):
    client = FakeClient(json.dumps(ANSWER))
    agent = Agent(config, client, retriever=FakeRetriever([RESULT, RESULT]))
    handle_command(Session(config, agent), "/json Who paid Delta?")

    out = capsys.readouterr().out
    assert client.formats_seen == [ANSWER_SCHEMA]
    assert json.dumps(ANSWER, indent=2) in out
    assert "Sources:\n  [1] leak/mail.eml" in out


def test_json_command_warns_about_unknown_excerpts(config, capsys):
    client = FakeClient(json.dumps(ANSWER))
    handle_command(Session(config, Agent(config, client, retriever=FakeRetriever([RESULT]))), "/json Who paid Delta?")
    assert "cites excerpts that do not exist: [2]" in capsys.readouterr().err


def test_json_command_shows_an_answer_it_cannot_read(config, capsys):
    handle_command(Session(config, Agent(config, FakeClient("Orion paid Delta."))), "/json Who paid Delta?")
    err = capsys.readouterr().err
    assert "the answer is not valid JSON" in err and "Orion paid Delta." in err


def test_json_command_needs_a_question(config, client, capsys):
    handle_command(Session(config, Agent(config, client)), "/json")
    assert "Usage: /json QUESTION" in capsys.readouterr().err
    assert client.chats == []
