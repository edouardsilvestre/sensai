import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sensai.core.guardrails.rules import (
    DEFAULT_RULES,
    GuardrailError,
    Guardrails,
    load_rules,
    model_judge,
    parse_rules,
    parse_verdict,
)
from sensai.core.llm.ollama import ToolCall

SOURCE = {"name": "source", "terms": ["Claire Benoît", "PHOENIX"], "on": ["output", "tools"], "replacement": "[SOURCE]"}


def guard(*rules: dict, judge=None) -> Guardrails:
    return Guardrails(parse_rules(list(rules)), judge)


def stream(guardrails: Guardrails, chunks: list[str]) -> tuple[str, object]:
    output = guardrails.output_filter()
    shown = "".join(output.feed(chunk) for chunk in chunks) + output.close()
    return shown, output


def test_terms_ignore_case_accents_and_spaces():
    checked = guard(SOURCE).check("output", "Selon claire  BENOIT, le dossier Phoenix est clos.")
    assert checked.text == "Selon [SOURCE], le dossier [SOURCE] est clos."
    assert checked.hits[0].count == 2


def test_terms_only_match_whole_words():
    assert guard({"name": "x", "terms": ["Alvor"]}).check("output", "Alvorada SA").text == "Alvorada SA"


def test_detectors_are_redacted_with_their_kind():
    checked = guard({"name": "pii", "detectors": ["email", "iban"]}).check(
        "output", "Mail marc@alvor.com, IBAN FR76 3000 6000 0112 3456 7890 189."
    )
    assert checked.text == "Mail [EMAIL], IBAN [IBAN]."
    assert checked.hits[0].found == "EMAIL, IBAN"


def test_rules_only_apply_to_their_stage():
    assert guard(SOURCE).check("input", "Claire Benoit").text == "Claire Benoit"


def test_block_rule_stops_the_check():
    rule = {"name": "home", "pattern": r"adresse (personnelle|du domicile)", "on": ["input"], "action": "block"}
    checked = guard(rule).check_input("Trouve l'adresse personnelle de Marc Vidal")
    assert checked.blocked.rule == "home"


def test_warn_rule_keeps_the_text():
    checked = guard({"name": "bank", "detectors": ["iban"], "action": "warn"}).check("output", "FR7630006000011234567890189")
    assert checked.text == "FR7630006000011234567890189" and checked.hits[0].action == "warn"


def test_tool_call_arguments_are_redacted():
    call, checked = guard(SOURCE).check_tool_call(ToolCall("web_search", {"query": "Claire Benoit Alvor", "max": 3}))
    assert call == ToolCall("web_search", {"query": "[SOURCE] Alvor", "max": 3})
    assert checked.blocked is None


def test_default_rules_block_personal_data_in_web_searches_only():
    guardrails = Guardrails(parse_rules(DEFAULT_RULES))
    _, search = guardrails.check_tool_call(ToolCall("web_search", {"query": "FR76 3000 6000 0112 3456 7890 189"}))
    _, code = guardrails.check_tool_call(ToolCall("run_python", {"code": "iban = 'FR76 3000 6000 0112 3456 7890 189'"}))
    assert search.blocked.found == "IBAN"
    assert code.blocked is None


def test_judge_blocks_with_its_reason():
    calls = []

    def judge(policy, message):
        calls.append((policy, message))
        return "asks for a private address"

    rule = {"name": "private-life", "judge": "finding a private address", "on": ["input"], "action": "block"}
    checked = guard(rule, judge=judge).check_input("Where does Marc Vidal live?")
    assert checked.blocked.found == "asks for a private address"
    assert calls == [("finding a private address", "Where does Marc Vidal live?")]


def test_judge_allows():
    rule = {"name": "private-life", "judge": "finding a private address", "on": ["input"], "action": "block"}
    assert guard(rule, judge=lambda policy, message: None).check_input("Who signed?").blocked is None


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("ALLOW", None),
        ("allow.", None),
        ("BLOCK: asks for a home address", "asks for a home address"),
        ("BLOCK", "the message breaks the policy"),
        ("I think it is fine", "the guardrail could not understand the check of the model"),
        ("", "the guardrail could not understand the check of the model"),
    ],
)
def test_unclear_verdicts_block(answer, expected):
    assert parse_verdict(answer) == expected


def test_model_judge_asks_the_chat_model(config, client):
    client.summary = "BLOCK: tracking a person"
    reason = model_judge(client, config)("tracking someone", "Follow Marc Vidal")
    assert reason == "tracking a person"
    prompt = client.summaries[0][0]["content"]
    assert "tracking someone" in prompt and "Follow Marc Vidal" in prompt


def test_split_term_is_redacted_in_a_stream():
    shown, output = stream(guard(SOURCE), ["Selon Claire Be", "noit, le virement", " a été validé."])
    assert shown == "Selon [SOURCE], le virement a été validé."
    assert output.hits["source"].count == 1


def test_long_stream_is_shown_before_the_end():
    output = guard(SOURCE).output_filter()
    shown = output.feed("mot " * 50)
    assert shown and shown.endswith(" ") and len(shown) < 200


def test_output_block_stops_the_stream():
    rule = {"name": "iban", "detectors": ["iban"], "action": "block"}
    shown, output = stream(guard(rule), ["Le compte est ", "FR76 3000 6000 0112 3456 7890 189", " et la suite"])
    assert shown == "Le compte est "
    assert output.blocked.found == "IBAN"


def test_stream_without_rules_is_unchanged():
    assert stream(Guardrails(), ["Hello", " there"])[0] == "Hello there"


def test_missing_file_gives_the_default_rules(tmp_path):
    assert [rule.name for rule in load_rules(tmp_path / "guardrails.json")] == ["personal-data", "published-personal-data"]


def test_rules_file_replaces_the_defaults(tmp_path):
    path = tmp_path / "guardrails.json"
    path.write_text(json.dumps({"rules": [SOURCE]}), encoding="utf-8")
    assert [rule.name for rule in load_rules(path)] == ["source"]


def test_example_file_is_valid():
    assert len(load_rules(Path(__file__).parent.parent / "guardrails.example.json")) == 5


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"terms": ["x"]}, "needs a name"),
        ({"name": "x"}, "exactly one of"),
        ({"name": "x", "terms": ["a"], "pattern": "b"}, "exactly one of"),
        ({"name": "x", "terms": []}, "'terms' must be"),
        ({"name": "x", "pattern": "("}, "invalid pattern"),
        ({"name": "x", "detectors": ["passport"]}, "'detectors' must list"),
        ({"name": "x", "terms": ["a"], "on": ["memory"]}, "'on' must list"),
        ({"name": "x", "terms": ["a"], "action": "delete"}, "'action' must be"),
        ({"name": "x", "judge": "doxxing", "on": ["output"], "action": "block"}, "judge rule only blocks"),
    ],
)
def test_invalid_rules_are_rejected(rule, message):
    with pytest.raises(GuardrailError, match=message):
        parse_rules([rule])


def test_duplicate_names_are_rejected():
    with pytest.raises(GuardrailError, match="unique: source"):
        parse_rules([SOURCE, SOURCE])


def test_unreadable_file_is_an_error(tmp_path):
    path = tmp_path / "guardrails.json"
    path.write_text("{rules: [", encoding="utf-8")
    with pytest.raises(GuardrailError, match="Cannot read the guardrails"):
        load_rules(path)


def test_the_same_rules_have_the_same_fingerprint():
    assert guard(SOURCE).fingerprint() == guard(dict(SOURCE)).fingerprint()


@pytest.mark.parametrize(
    "changed",
    [
        {**SOURCE, "terms": ["Claire Benoît", "PHOENIX", "Marc Vidal"]},
        {**SOURCE, "on": ["output"]},
        {**SOURCE, "action": "block"},
        {**SOURCE, "replacement": "[X]"},
    ],
)
def test_a_changed_rule_changes_the_fingerprint(changed):
    assert guard(changed).fingerprint() != guard(SOURCE).fingerprint()


def test_an_added_rule_changes_the_fingerprint():
    assert guard(SOURCE, *DEFAULT_RULES).fingerprint() != guard(SOURCE).fingerprint()


def test_the_fingerprint_is_the_same_in_every_process():
    code = (
        "from sensai.core.guardrails.rules import DEFAULT_RULES, Guardrails, parse_rules; "
        f"print(Guardrails(parse_rules(DEFAULT_RULES + [{SOURCE!r}])).fingerprint())"
    )
    prints = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(prints) == 1
