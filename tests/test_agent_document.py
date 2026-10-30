import json

from sensai.core.agent import Agent
from sensai.core.document.document import Edit
from sensai.core.events import DocumentEvent, DocumentFailedEvent, DoneEvent, GuardrailEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError
from sensai.core.memory.conversation import DOCUMENT_HEADER

SOURCE = {"name": "source", "terms": ["Claire Benoit"], "on": ["output"], "replacement": "[SOURCE]"}


def edits(*items: dict) -> str:
    return json.dumps({"edits": list(items)})


def of_type(events, kind) -> list:
    return [event for event in events if isinstance(event, kind)]


def test_the_document_is_updated_after_the_answer(config, client):
    client.edits = edits({"op": "append", "section": "Timeline", "text": "- 4 April 2023: transfer approved"})
    agent = Agent(config, client)
    events = list(agent.run("When was the transfer approved?"))
    assert isinstance(events[-1], DocumentEvent)
    assert events.index(of_type(events, DoneEvent)[0]) < events.index(events[-1])
    assert events[-1].report.applied == [Edit("append", "Timeline", "- 4 April 2023: transfer approved")]
    assert agent.memory.document.section("Timeline").body == "- 4 April 2023: transfer approved"


def test_the_editor_gets_the_exchange_and_not_the_chat_history(config, client):
    agent = Agent(config, client)
    list(agent.run("First question"))
    client.answer = "Second answer"
    list(agent.run("Second question"))
    prompt = client.edit_requests[-1][0]["content"]
    assert "Journalist: Second question\nAssistant: Second answer" in prompt
    assert "First question" not in prompt


def test_the_document_is_sent_to_the_model_and_kept_out_of_the_history(config, client):
    client.edits = edits({"op": "append", "section": "People and companies", "text": "- Marc Vidal, CFO"})
    agent = Agent(config, client)
    list(agent.run("Who is Marc Vidal?"))
    list(agent.run("And then?"))
    system = client.chats[-1][0][0]["content"]
    assert f"{DOCUMENT_HEADER}\n# Investigation" in system and "- Marc Vidal, CFO" in system
    assert all("Marc Vidal, CFO" not in turn.assistant for turn in agent.memory.turns)


def test_an_empty_document_is_not_sent(config, client):
    agent = Agent(config, client)
    list(agent.run("Hi"))
    assert DOCUMENT_HEADER not in str(client.chats[-1][0])


def test_rejected_edits_are_reported_and_the_valid_ones_applied(config, client):
    client.edits = edits(
        {"op": "append", "section": "Suspects", "text": "- Marc Vidal"},
        {"op": "append", "section": "Timeline", "text": "- 4 April"},
    )
    agent = Agent(config, client)
    report = of_type(list(agent.run("When?")), DocumentEvent)[0].report
    assert [edit.section for edit in report.applied] == ["Timeline"]
    assert report.rejected[0].reason.startswith("no section 'Suspects'")


def test_a_failed_update_keeps_the_answer_and_the_document(config, client):
    client.edits = "not json"
    agent = Agent(config, client)
    events = list(agent.run("When?"))
    assert "not valid JSON" in of_type(events, DocumentFailedEvent)[0].message
    assert agent.memory.turns and agent.memory.document.is_empty


def test_an_ollama_error_during_the_update_is_reported(config, client):
    def editor(document, user, answer):
        raise LLMError("Lost connection to Ollama")

    events = list(Agent(config, client, editor=editor).run("When?"))
    assert of_type(events, DocumentFailedEvent)[0].message == "Lost connection to Ollama"


def test_nothing_is_reported_when_nothing_changes(config, client):
    events = list(Agent(config, client).run("Hi"))
    assert not of_type(events, DocumentEvent) and not of_type(events, DocumentFailedEvent)
    assert len(client.edit_requests) == 1


def test_the_feature_off_makes_no_call_and_sends_no_document(config, client):
    client.edits = edits({"op": "append", "section": "Timeline", "text": "- 4 April"})
    agent = Agent(config, client)
    list(agent.run("When?"))
    config.features["document"] = False
    list(agent.run("And then?"))
    assert len(client.edit_requests) == 1
    assert DOCUMENT_HEADER not in str(client.chats[-1][0])


def test_the_guardrails_redact_the_document_too(config, client):
    client.edits = edits({"op": "append", "section": "Timeline", "text": "- Claire Benoit refused the transfer"})
    agent = Agent(config, client, guardrails=Guardrails(parse_rules([SOURCE])))
    events = list(agent.run("Who refused?"))
    assert agent.memory.document.section("Timeline").body == "- [SOURCE] refused the transfer"
    assert of_type(events, GuardrailEvent)[-1].hit.rule == "source"


def test_an_edit_blocked_by_a_guardrail_is_not_written(config, client):
    rule = {"name": "iban", "detectors": ["iban"], "on": ["input", "output"], "action": "block"}
    client.edits = edits({"op": "append", "section": "Timeline", "text": "- FR7630006000011234567890189 got the money"})
    agent = Agent(config, client, guardrails=Guardrails(parse_rules([rule])))
    report = of_type(list(agent.run("Which account?")), DocumentEvent)[0].report
    assert report.rejected[0].reason == "blocked by the guardrail iban (IBAN)"
    assert agent.memory.document.is_empty


def test_a_blocked_message_does_not_update_the_document(config, client):
    rule = {"name": "private", "pattern": "adresse personnelle", "on": ["input"], "action": "block"}
    agent = Agent(config, client, guardrails=Guardrails(parse_rules([rule])))
    list(agent.run("Donne son adresse personnelle"))
    assert client.edit_requests == []


def test_an_answer_stopped_by_a_guardrail_does_not_update_the_document(config, client):
    client.answer = "Le compte FR7630006000011234567890189 a reçu les fonds."
    rule = {"name": "iban", "detectors": ["iban"], "action": "block"}
    list(Agent(config, client, guardrails=Guardrails(parse_rules([rule]))).run("Quel compte ?"))
    assert client.edit_requests == []
