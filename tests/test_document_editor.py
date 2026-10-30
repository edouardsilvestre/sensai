import json

import pytest

from sensai.core.document.document import Document, DocumentError, Edit
from sensai.core.document.editor import EDITS_SCHEMA, build_editor_prompt, model_editor, parse_edits


def test_edits_are_parsed():
    reply = json.dumps({"edits": [
        {"op": "append", "section": "Timeline", "text": "- 4 April"},
        {"op": "replace", "section": "People", "find": "CFO", "text": "CEO"},
    ]})
    assert parse_edits(reply) == [Edit("append", "Timeline", "- 4 April"), Edit("replace", "People", "CEO", "CFO")]


def test_a_reply_in_a_code_block_is_read():
    assert parse_edits('```json\n{"edits": []}\n```') == []


@pytest.mark.parametrize(
    ("reply", "message"),
    [("Sure! Here are the edits", "not valid JSON"), ('{"changes": []}', "the reply is not"), ("[]", "not {")],
)
def test_a_reply_that_is_not_edits_is_an_error(reply, message):
    with pytest.raises(DocumentError, match=message):
        parse_edits(reply)


def test_a_malformed_edit_is_kept_so_that_it_is_rejected_with_the_reason():
    edits = parse_edits('{"edits": ["append", {"op": "append", "section": 3, "text": "x"}]}')
    assert edits == [Edit("", "", ""), Edit("append", "", "x")]
    report = Document().apply(edits)
    assert [rejected.reason for rejected in report.rejected][1] == "no section given"


def test_the_prompt_gives_the_sections_the_document_and_the_exchange():
    document = Document()
    document.apply([Edit("append", "Timeline", "- 4 April 2023: transfer approved")])
    prompt = build_editor_prompt(document, "Who signed the invoice?", "Nobody: it is missing.")
    assert "Sections: Summary, People and companies, Timeline, Open questions" in prompt
    assert "- 4 April 2023: transfer approved" in prompt
    assert "Journalist: Who signed the invoice?\nAssistant: Nobody: it is missing." in prompt


def test_the_model_answers_with_the_schema_in_a_call_of_its_own(config, client):
    client.edits = '{"edits": [{"op": "append", "section": "Open questions", "text": "- Who signed the invoice?"}]}'
    edits = model_editor(client, config)(Document(), "Who signed the invoice?", "Nobody knows yet.")
    assert edits == [Edit("append", "Open questions", "- Who signed the invoice?")]
    assert len(client.edit_requests) == 1 and client.summaries == [] and client.chats == []


def test_the_schema_only_allows_the_known_operations():
    assert EDITS_SCHEMA["properties"]["edits"]["items"]["properties"]["op"]["enum"] == [
        "append", "replace", "rewrite", "add_section"
    ]
