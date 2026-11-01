import json

import pytest

from sensai.core.agent import Agent
from sensai.core.persona import Persona
from sensai.core.prompts import (
    COMPARE_OPTIONS, Case, PromptError, PromptStore, PromptVersion, check, compare, diff, load_cases, score,
)

EDITOR = Persona("editor", "Editor", "an editor.", "clear.", "write drafts.")


@pytest.fixture
def store(tmp_path) -> PromptStore:
    return PromptStore(tmp_path / "prompts")


def test_first_snapshot_is_version_1_and_active(store):
    version = store.snapshot("editor", "Be clear.")
    assert (version.number, version.prompt) == (1, "Be clear.")
    assert store.active("editor") == version and version.created_at


def test_same_prompt_makes_no_new_version(store):
    store.snapshot("editor", "Be clear.")
    assert store.snapshot("editor", "Be clear.") is None
    assert len(store.history("editor").versions) == 1


def test_changed_prompt_becomes_the_active_version(store):
    store.snapshot("editor", "Be clear.")
    store.snapshot("editor", "Be short.")
    store.activate("editor", 1)
    assert store.snapshot("editor", "Be precise.").number == 3
    assert store.active("editor").prompt == "Be precise."


def test_rollback_survives_a_restart(store):
    store.snapshot("editor", "Be clear.")
    store.snapshot("editor", "Be short.")
    store.activate("editor", 1)

    restarted = PromptStore(store.directory)
    assert restarted.snapshot("editor", "Be short.") is None
    assert restarted.active("editor").prompt == "Be clear."


def test_unknown_version_is_refused(store):
    store.snapshot("editor", "Be clear.")
    with pytest.raises(PromptError, match="No version 3 of editor: it has 1"):
        store.activate("editor", 3)


def test_persona_without_versions(store):
    assert store.active("legal") is None and store.history("legal").versions == []


def test_invalid_file_names_the_file(store):
    store.directory.mkdir()
    (store.directory / "editor.json").write_text('{"versions": [], "active": 1}', encoding="utf-8")
    with pytest.raises(PromptError, match="editor.json"):
        store.active("editor")


def test_ids_cannot_leave_the_folder(store):
    with pytest.raises(PromptError, match="Invalid persona id"):
        store.snapshot("../editor", "Be clear.")


def test_diff_shows_the_changed_lines():
    text = diff(PromptVersion(1, "You are Ana.\nBe clear.", ""), PromptVersion(2, "You are Ana.\nBe short.", ""))
    assert "--- version 1" in text and "-Be clear." in text and "+Be short." in text and " You are Ana." in text


def test_agent_uses_the_active_version(config, client, store):
    store.snapshot("editor", EDITOR.system_prompt())
    store.snapshot("editor", "Old prompt kept on purpose.")
    store.activate("editor", 2)
    agent = Agent(config, client, persona=EDITOR, prompts=store)
    assert agent.prompt_version == 2
    assert agent.messages[0] == {"role": "system", "content": "Old prompt kept on purpose."}


def test_agent_uses_the_file_when_the_feature_is_off(config, client, store):
    store.snapshot("editor", "Old prompt kept on purpose.")
    config.features["prompt_versions"] = False
    agent = Agent(config, client, persona=EDITOR, prompts=store)
    assert agent.prompt_version is None
    assert agent.messages[0]["content"] == EDITOR.system_prompt()


def write_cases(path, cases):
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return path


def test_cases_are_loaded(tmp_path):
    path = write_cases(tmp_path / "cases.json", [{"question": "Who?", "must_contain": ["Vidal"]}, {"question": "When?"}])
    assert load_cases(path) == [Case("Who?", ["Vidal"]), Case("When?")]


@pytest.mark.parametrize("cases", [[], [{"must_contain": ["a"]}], [{"question": "Who?", "must_contain": "Vidal"}]])
def test_invalid_cases_are_refused(tmp_path, cases):
    with pytest.raises(PromptError, match="cases.json"):
        load_cases(write_cases(tmp_path / "cases.json", cases))


def test_missing_cases_file_points_to_the_example(tmp_path):
    with pytest.raises(PromptError, match="prompt_cases.example.json"):
        load_cases(tmp_path / "cases.json")


def test_check_ignores_the_case():
    case = Case("Who?", must_contain=["Marc Vidal", "[1]"], must_not_contain=["sources say"])
    assert check(case, "MARC VIDAL signed [1].") == []
    assert check(case, "Sources say someone signed.") == ['missing "Marc Vidal"', 'missing "[1]"', 'contains "sources say"']


def test_score_asks_each_case_with_the_prompt_alone(config, client):
    client.summary = "Marc Vidal signed."
    ticks = iter([0.0, 1.5, 2.0, 2.5])
    cases = [Case("Who signed?", ["Vidal"]), Case("When?", ["2023"])]
    result = score(client, config, PromptVersion(2, "Be clear.", ""), cases, clock=lambda: next(ticks))

    assert client.summaries[0] == [
        {"role": "system", "content": "Be clear."}, {"role": "user", "content": "Who signed?"},
    ]
    assert (result.version, result.passed, result.seconds, result.tokens) == (2, 1, 1.0, 320)
    assert result.results[1].failed == ['missing "2023"']
    assert result.summary() == {"version": 2, "passed": 1, "cases": 2, "seconds": 1.0, "tokens": 320}


def test_score_fixes_the_sampling(config, client):
    options = []
    client.chat = lambda model, messages, opts=None, think=None, usage=None: options.append(opts) or ""
    score(client, config, PromptVersion(1, "Be clear.", ""), [Case("Who?")])
    assert options == [{"num_ctx": config.num_ctx, **COMPARE_OPTIONS}]


def test_compare_scores_every_version_on_the_same_cases(config, client):
    write_cases(config.prompt_cases_path, [{"question": "Who?"}, {"question": "When?"}])
    versions = [PromptVersion(1, "Be clear.", ""), PromptVersion(2, "Be short.", "")]
    scores = compare(client, config, versions)

    assert [result.version for result in scores] == [1, 2]
    assert [messages[0]["content"] for messages in client.summaries] == ["Be clear.", "Be clear.", "Be short.", "Be short."]


def test_compare_without_cases_asks_nothing(config, client):
    with pytest.raises(PromptError, match="No test cases"):
        compare(client, config, [PromptVersion(1, "Be clear.", "")])
    assert client.summaries == []
