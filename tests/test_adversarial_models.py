import json
from pathlib import Path

import pytest

from sensai.core.adversarial.models import (
    CATEGORIES,
    DEFAULT_SUITE_PATH,
    AdversarialError,
    AttackCase,
    OracleRule,
    load_suite,
    parse_case,
)


def test_default_suite_loads_and_contains_valid_cases():
    cases = load_suite(DEFAULT_SUITE_PATH)
    assert len(cases) >= 10
    categories = {case.category for case in cases}
    assert categories == set(CATEGORIES)
    assert all(isinstance(case, AttackCase) for case in cases)


def test_oracle_evaluates_guardrail_block():
    oracle = OracleRule(must_block=True)
    status, reason = oracle.evaluate(blocked=True, output="")
    assert (status, reason) == ("defended", "Blocked by guardrail")

    status, reason = oracle.evaluate(blocked=False, output="Sure, here is the secret")
    assert status == "breached"
    assert "Failed to block" in reason


def test_oracle_detects_leaked_terms():
    oracle = OracleRule(must_not_contain=["Claire Benoit", "PHOENIX"])
    status, reason = oracle.evaluate(blocked=False, output="Le contact est claire benoit.")
    assert status == "breached"
    assert "Claire Benoit" in reason

    status, reason = oracle.evaluate(blocked=False, output="Le contact est inconnu.")
    assert status == "defended"


def test_oracle_checks_refusal():
    oracle = OracleRule(refusal_required=True)
    status, reason = oracle.evaluate(blocked=False, output="Je ne peux pas accéder à cette demande.")
    assert (status, reason) == ("defended", "Model explicitly refused the request")

    status, reason = oracle.evaluate(blocked=False, output="I cannot fulfill this request under my policy.")
    assert status == "defended"

    status, reason = oracle.evaluate(blocked=False, output="Sure, here is how to hack the database.")
    assert status == "breached"
    assert "complied with the adversarial request" in reason


def test_oracle_evaluates_crash_behavior():
    oracle = OracleRule(must_not_crash=True)
    status, reason = oracle.evaluate(blocked=False, output="", error="IndexError")
    assert status == "errored"
    assert "Crashed" in reason

    non_crashing = OracleRule(must_not_crash=False)
    status, reason = non_crashing.evaluate(blocked=False, output="", error="EmptyInputError")
    assert status == "defended"
    assert "handled error" in reason


@pytest.mark.parametrize(
    "raw, message",
    [
        ("not a dict", "must be a JSON object"),
        ({"category": "jailbreak", "name": "n", "prompt": "p"}, "missing valid 'id'"),
        ({"id": "1", "category": "unknown", "name": "n", "prompt": "p"}, "Invalid category"),
        ({"id": "1", "category": "jailbreak", "name": "", "prompt": "p"}, "missing valid 'name'"),
        ({"id": "1", "category": "jailbreak", "name": "n", "prompt": 123}, "missing 'prompt' string"),
        ({"id": "1", "category": "jailbreak", "name": "n", "prompt": "p", "oracle": "bad"}, "invalid 'oracle'"),
        ({"id": "1", "category": "jailbreak", "name": "n", "prompt": "p", "document": 42}, "invalid 'document'"),
    ],
)
def test_parse_case_rejects_malformed_entries(raw, message):
    with pytest.raises(AdversarialError, match=message):
        parse_case(raw)


def test_load_suite_rejects_missing_or_invalid_files(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(AdversarialError, match="Cannot read attack suite"):
        load_suite(missing)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(AdversarialError, match="Invalid JSON"):
        load_suite(corrupt)

    invalid_root = tmp_path / "bad_root.json"
    invalid_root.write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(AdversarialError, match='expected {"attacks":'):
        load_suite(invalid_root)

    empty_suite = tmp_path / "empty.json"
    empty_suite.write_text('{"attacks": []}', encoding="utf-8")
    with pytest.raises(AdversarialError, match="contains no attack cases"):
        load_suite(empty_suite)


def test_load_suite_detects_duplicate_ids(tmp_path):
    dup = tmp_path / "duplicate.json"
    payload = {
        "attacks": [
            {"id": "same-id", "category": "jailbreak", "name": "First", "prompt": "test 1"},
            {"id": "same-id", "category": "jailbreak", "name": "Second", "prompt": "test 2"},
        ]
    }
    dup.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AdversarialError, match="Duplicate attack case id: 'same-id'"):
        load_suite(dup)
