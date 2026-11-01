import json

import pytest

from sensai.core.evaluation.retrieval import evaluate_retrieval, matches, ranks, score
from sensai.core.evaluation.runner import Case, EvalError, load_cases
from sensai.core.rag.index import DocumentIndex
from sensai.core.rag.models import SearchResult
from sensai.core.rag.reranker import RerankError, Reranker
from sensai.interfaces.cli.main import main


def found(*sources: str, relevance: float | None = None) -> list[SearchResult]:
    return [SearchResult(n, "text", "", source, "", 0.5, relevance) for n, source in enumerate(sources, start=1)]


def test_a_source_is_named_by_its_path_or_its_end():
    assert matches("leak/mails/mail_042.eml", "mail_042.eml")
    assert matches("leak/mails/mail_042.eml", "mails/mail_042.eml")
    assert not matches("leak/mails/old_mail_042.eml", "mail_042.eml")


def test_each_document_that_answers_is_ranked_at_its_first_chunk():
    case = Case("q", sources=["a.txt", "b.txt", "c.txt"])
    assert ranks(found("x.txt", "b.txt", "a.txt", "b.txt"), case) == [3, 2]


def test_the_score_of_a_ranking():
    case = Case("q", sources=["a.txt", "b.txt"])
    assert score("s", [case], [found("a.txt", "x.txt", "b.txt")], k=2).recall == 0.5
    assert score("s", [case], [found("b.txt", "a.txt", "x.txt")], k=2).ndcg == pytest.approx(1.0)
    lower = score("s", [case], [found("x.txt", "a.txt", "b.txt")], k=2).ndcg
    assert 0 < lower < score("s", [case], [found("a.txt", "x.txt", "b.txt")], k=2).ndcg < 1
    # A second chunk of the same document is not a second answer.
    assert score("s", [case], [found("a.txt", "a.txt")], k=2).ndcg < 1


def test_the_score_of_a_step():
    cases = [Case("q1", sources=["a.txt"]), Case("q2", sources=["b.txt"]), Case("no answer")]
    rankings = [found("a.txt", "x.txt"), found("x.txt", "y.txt"), found("z.txt")]
    result = score("step", cases, rankings, k=2)
    assert (result.recall, result.precision, result.noise) == (0.5, 0.25, 1.0)
    assert score("step", [Case("no answer")], [[]], k=2).recall is None


def test_the_questions_file_names_the_sources(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"cases": [{"question": "Qui ?", "sources": [" mail_001.eml "]}]}), encoding="utf-8")
    assert load_cases(path) == [Case("Qui ?", [], ["mail_001.eml"])]
    path.write_text(json.dumps({"cases": [{"question": "Qui ?", "sources": "mail_001.eml"}]}), encoding="utf-8")
    with pytest.raises(EvalError, match="sources of question 1 are not texts"):
        load_cases(path)


@pytest.fixture
def leak(tmp_path, config, client):
    folder = tmp_path / "leak"
    folder.mkdir()
    (folder / "march.txt").write_text("Alice approved the transfer of March to Delta.", encoding="utf-8")
    (folder / "may.txt").write_text("Bob approved the transfer of May to Omega.", encoding="utf-8")
    (folder / "budget.txt").write_text("The board voted the annual budget.", encoding="utf-8")
    index = DocumentIndex.open(config, client)
    index.ingest([folder])
    yield index
    index.close()


def test_each_step_and_threshold_is_measured_and_the_config_is_given_back(leak, config):
    # Only the email of March answers, and only the question on March.
    leak.reranker = Reranker("fake", scorer=lambda pairs: [5.0 if "March" in q and "March" in p else -5.0 for q, p in pairs])
    cases = [Case("Who approved the transfer of March?", sources=["march.txt"]), Case("Who owns a yacht?")]
    config.rerank_threshold = 25
    scores = {row.name: row for row in evaluate_retrieval(leak, config, cases)}

    assert list(scores) == [
        "vector (R1)", "+ hybrid", "+ rerank", "+ threshold 10%", "+ threshold 20%", "+ threshold 25%",
        "+ threshold 30%", "+ threshold 40%", "+ threshold 50%",
    ]
    assert scores["+ rerank"].ndcg == 1.0
    assert (scores["+ rerank"].noise, scores["+ threshold 25%"].noise) == (3.0, 0.0)
    assert scores["+ threshold 25%"].precision == 1.0
    assert leak.config is config


def test_a_reranker_that_cannot_run_fails_the_evaluation(leak, config):
    def score(pairs):
        raise RerankError("Cannot load the rerank model 'fake'")

    leak.reranker = Reranker("fake", scorer=score)
    with pytest.raises(RerankError, match="Cannot load the rerank model 'fake'"):
        evaluate_retrieval(leak, config, [Case("Who approved the transfer of March?", sources=["march.txt"])])
    assert leak.config is config


def test_the_cli_prints_one_table_and_marks_the_threshold(tmp_path, monkeypatch, capsys, leak, config, client):
    leak.close()
    questions = tmp_path / "eval.json"
    questions.write_text(json.dumps({"cases": [{"question": "the budget?", "sources": ["budget.txt"]}]}), encoding="utf-8")
    monkeypatch.setattr("sensai.interfaces.cli.main.load_config", lambda path: config)
    monkeypatch.setattr("sensai.interfaces.cli.main.OllamaClient", lambda *args: client)

    assert main(["eval", str(questions), "--retrieval"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Searching 3 documents for 1 question (1 answered by the documents)...\n")
    assert "Recall@3  NDCG@10  Precision  Noise" in out
    [marked] = [line for line in out.splitlines() if line.endswith("← SENSAI_RERANK_THRESHOLD")]
    assert marked.split()[:5] == ["+", "threshold", "20%", "100%", "1.00"]


def test_the_cli_refuses_retrieval_with_out_or_compare(tmp_path, capsys):
    assert main(["eval", str(tmp_path / "eval.json"), "--retrieval", "--out", "r.json"]) == 1
    assert "cannot be used with --out or --compare" in capsys.readouterr().err
