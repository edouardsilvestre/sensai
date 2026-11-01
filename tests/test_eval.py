import json

import pytest

from sensai.core.agent import Agent
from sensai.core.evaluation.runner import (
    Case,
    CaseResult,
    EvalError,
    contains,
    eval_config,
    load_cases,
    load_results,
    run_case,
    run_eval,
    save_results,
    summarize,
)
from sensai.core.llm.ollama import LLMError
from sensai.interfaces.cli.main import main

from test_agent_check import ANSWER, JudgeClient, Retriever

QUESTION = "Qui dirige Orion ?"


def write(path, data) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def make_agent(config, client) -> Agent:
    return Agent(eval_config(config), client, retriever=Retriever())


def test_cases_are_read_from_the_file(tmp_path):
    path = tmp_path / "eval.json"
    write(path, {"cases": [{"question": f" {QUESTION} ", "expected": [" Marc Vidal "]}, {"question": "Et Nordbank ?"}]})
    assert load_cases(path) == [Case(QUESTION, ["Marc Vidal"]), Case("Et Nordbank ?", [])]


@pytest.mark.parametrize(
    "data, message",
    [
        ({"questions": []}, 'expected {"cases": [...]}'),
        ({"cases": []}, "at least one question"),
        ({"cases": [{"question": " "}]}, "question 1 is empty"),
        ({"cases": ["Qui ?"]}, "question 1 is empty"),
        ({"cases": [{"question": "Qui ?", "expected": "Marc Vidal"}]}, "expected facts of question 1 are not texts"),
        ({"cases": [{"question": "Qui ?", "expected": [480000]}]}, "expected facts of question 1 are not texts"),
    ],
)
def test_an_invalid_cases_file_is_an_error(tmp_path, data, message):
    path = tmp_path / "eval.json"
    write(path, data)
    with pytest.raises(EvalError, match=message.replace("[", r"\[").replace("{", r"\{")):
        load_cases(path)


def test_a_missing_or_broken_cases_file_is_an_error(tmp_path):
    with pytest.raises(EvalError, match="Cannot read the questions"):
        load_cases(tmp_path / "missing.json")
    (tmp_path / "eval.json").write_text("{cases", encoding="utf-8")
    with pytest.raises(EvalError, match="Invalid questions file"):
        load_cases(tmp_path / "eval.json")


@pytest.mark.parametrize(
    "answer, expected, found",
    [
        ("Orion a reçu 480 000,00 EUR.", "480 000", True),
        ("Le virement date du 5 avril 2023.", "05/04/2023", True),
        ("Le virement date d'avril 2023.", "05/04/2023", False),
        ("La société est gérée par M. Marc Vidal.", "Marc Vidal", True),
        ("La société est gérée par M. Marc Vidalès.", "Marc Vidal", False),
        ("Le virement vient de NORDBANK SA.", "Nordbank", True),
        ("Les documents ne le disent pas.", "Nordbank", False),
    ],
)
def test_an_expected_fact_is_found_whatever_its_format(answer, expected, found):
    assert contains(answer, expected) is found


def test_the_eval_turns_off_what_would_make_answers_depend_on_each_other(config):
    config.features.update(cache=True, examples=True, fact_check=True, document=True)
    checked = eval_config(config)
    assert [checked.features[name] for name in ("cache", "examples", "fact_check", "document")] == [False] * 4
    assert config.features["cache"] is True


def test_a_case_is_answered_then_judged(config):
    client = JudgeClient("contradicted")
    result = run_case(make_agent(config, client), eval_config(config), Case(QUESTION, ["Marc Vidal", "480 000"]))

    assert (result.answer, result.found, result.error) == (ANSWER, ["Marc Vidal"], "")
    assert (result.faithfulness, result.relevance, result.coherence, result.unchecked) == (0.0, 5, 4, 0)
    assert [claim["verdict"] for claim in result.flagged] == ["contradicted"]
    assert "[1] statuts.pdf, p. 2:" in client.judged[0]


def test_a_failed_answer_is_recorded_and_not_judged(config):
    client = JudgeClient()

    def broken(*args, **kwargs):
        raise LLMError("Cannot reach Ollama")

    client.chat_stream = broken
    result = run_case(make_agent(config, client), eval_config(config), Case(QUESTION))
    assert (result.error, result.answer, client.judged) == ("Cannot reach Ollama", "", [])


def test_a_failed_judgement_keeps_the_answer(config):
    client = JudgeClient()
    client.broken = True
    result = run_case(make_agent(config, client), eval_config(config), Case(QUESTION))
    assert result.answer == ANSWER and result.error == "The answer could not be judged: Cannot reach Ollama"


def test_each_case_gets_a_new_agent(config):
    client = JudgeClient("supported", "supported")
    agents = []

    def fresh() -> Agent:
        agents.append(make_agent(config, client))
        return agents[-1]

    shown = []
    cases = [Case(QUESTION), Case("Et Nordbank ?")]
    results = run_eval(fresh, eval_config(config), cases, lambda n, r: shown.append(n))
    assert len({id(agent) for agent in agents}) == 2 and shown == [1, 2]
    assert [len(agent.memory.turns) for agent in agents] == [1, 1] and len(results) == 2


def test_the_summary_averages_what_was_measured():
    summary = summarize([
        CaseResult("a", "x", 4.0, ["Marc Vidal", "480 000"], ["Marc Vidal"], 1.0, 5, 4),
        CaseResult("b", "y", 8.0, ["Nordbank"], ["Nordbank"], 0.5, 3, 4, [{"text": "y"}]),
        CaseResult("c", "z", 6.0, [], [], None, None, None),
        CaseResult("d", error="Cannot reach Ollama"),
    ])
    assert (summary.cases, summary.errors, summary.flagged) == (4, 1, 1)
    assert summary.expected == pytest.approx(2 / 3)
    assert (summary.faithfulness, summary.relevance, summary.coherence, summary.seconds) == (0.75, 4.0, 4.0, 6.0)


def test_an_empty_summary_has_no_averages():
    summary = summarize([CaseResult("a", error="Cannot reach Ollama")])
    assert (summary.expected, summary.faithfulness, summary.seconds) == (None, None, None)


def test_results_are_saved_with_their_configuration_and_read_back(config, tmp_path):
    config.judge_model = "qwen3:1.7b"
    results = [CaseResult(QUESTION, ANSWER, 4.2, ["Marc Vidal"], ["Marc Vidal"], 1.0, 5, 4)]
    path = tmp_path / "out" / "results.json"
    save_results(path, config, results)

    data, read = load_results(path)
    assert (data["model"], data["judge_model"], data["summary"]["faithfulness"]) == ("chat-model", "qwen3:1.7b", 1.0)
    assert data["features"] == config.features and read == results


def test_unreadable_results_are_an_error(tmp_path):
    (tmp_path / "results.json").write_text('{"cases": [{"answer": "x"}]}', encoding="utf-8")
    with pytest.raises(EvalError, match="Cannot read the results"):
        load_results(tmp_path / "results.json")


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SENSAI_FEATURE_TOOLS", "false")
    client = JudgeClient("supported", "supported", "supported", "supported")
    monkeypatch.setattr("sensai.interfaces.cli.main.OllamaClient", lambda host, key="": client)
    cases = write(tmp_path / "eval.json", {"cases": [{"question": QUESTION, "expected": ["Marc Vidal", "Nordbank"]}]})
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    return client, cases, str(env)


def test_the_eval_command_prints_each_case_and_the_summary(cli, tmp_path, capsys):
    client, cases, env = cli
    assert main(["-e", env, "eval", cases, "--out", str(tmp_path / "first.json")]) == 0
    out = capsys.readouterr().out
    assert f"[1/1] {QUESTION}" in out and "expected 1/2" in out and "missing from the answer: Nordbank" in out
    assert "1 question: expected facts 50%" in out and "Results written to" in out
    assert json.loads((tmp_path / "first.json").read_text())["cases"][0]["found"] == ["Marc Vidal"]


def test_the_eval_command_compares_with_a_former_run(cli, tmp_path, capsys):
    client, cases, env = cli
    main(["-e", env, "eval", cases, "--out", str(tmp_path / "first.json")])
    client.answer = "Marc Vidal dirige Orion [1], payée par Nordbank."
    capsys.readouterr()

    assert main(["-e", env, "eval", cases, "--compare", str(tmp_path / "first.json")]) == 0

    out = capsys.readouterr().out
    assert "Compared with" in out and "expected: 50% → 100% (+50)" in out


def test_the_eval_command_stops_on_an_invalid_file(cli, tmp_path, capsys):
    _, _, env = cli
    assert main(["-e", env, "eval", write(tmp_path / "bad.json", {"cases": []})]) == 1
    assert "at least one question" in capsys.readouterr().err
