"""
Scores the document search alone, without the chat model, on questions whose answering documents are known:
each step of the search with the steps before it, then the threshold at several values.
"""

import math
from dataclasses import dataclass, replace

from sensai.core.config import Config
from sensai.core.evaluation.runner import Case
from sensai.core.rag.index import DocumentIndex
from sensai.core.rag.models import SearchResult
from sensai.core.rag.reranker import RerankError

STEPS = {
    "vector (R1)": {"hybrid": False, "rerank": False},
    "+ hybrid": {"hybrid": True, "rerank": False},
    "+ rerank": {"hybrid": True, "rerank": True},
}
THRESHOLDS = (10, 20, 30, 40, 50)
NDCG_K = 10


@dataclass
class Score:
    name: str
    # Share of the documents that answer found among the excerpts sent to the model.
    recall: float | None
    # 1 when the documents that answer come first among the 10 best, less the lower they are.
    ndcg: float | None
    # Share of the excerpts sent that come from a document that answers.
    precision: float | None
    # Excerpts sent for a question the documents do not answer.
    noise: float | None


def matches(source: str, wanted: str) -> bool:
    """`wanted` names a document by its path in the index, or by its end, like "mail_042.eml"."""
    return source == wanted or source.endswith("/" + wanted)


def ranks(ranking: list[SearchResult], case: Case) -> list[int]:
    """The rank of each document that answers, at its first chunk: two chunks of it are not two answers."""
    found = [next((n for n, r in enumerate(ranking, start=1) if matches(r.source, w)), 0) for w in case.sources]
    return [rank for rank in found if rank]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score(name: str, cases: list[Case], rankings: list[list[SearchResult]], k: int) -> Score:
    answered = [(case, ranking) for case, ranking in zip(cases, rankings) if case.sources]
    unanswered = [ranking for case, ranking in zip(cases, rankings) if not case.sources]

    def ndcg(case: Case, ranking: list[SearchResult]) -> float:
        gain = sum(1 / math.log2(rank + 1) for rank in ranks(ranking, case) if rank <= NDCG_K)
        return gain / sum(1 / math.log2(rank + 1) for rank in range(1, min(len(case.sources), NDCG_K) + 1))

    def precision(case: Case, sent: list[SearchResult]) -> float:
        return sum(any(matches(r.source, w) for w in case.sources) for r in sent) / len(sent)

    return Score(
        name,
        recall=_mean([sum(rank <= k for rank in ranks(r, case)) / len(case.sources) for case, r in answered]),
        ndcg=_mean([ndcg(case, r) for case, r in answered]),
        precision=_mean([precision(case, r[:k]) for case, r in answered if r]),
        noise=_mean([len(r[:k]) for r in unanswered]),
    )


def evaluate_retrieval(index: DocumentIndex, config: Config, cases: list[Case]) -> list[Score]:
    """
    The neighbors are left out: they change the text of the excerpts, not which chunks were found.
    The reranker runs once with no threshold, and each threshold is applied to the relevance it gave.
    """
    k = config.rag_top_k
    rankings = {}
    try:
        for name, features in STEPS.items():
            index.config = replace(
                config, features={**config.features, **features}, rerank_threshold=0, rag_neighbors=0
            )
            rankings[name] = [index.search(case.question, max(k, NDCG_K)) for case in cases]
            # Without the reranker, its rows would be the scores of the search before it.
            if index.rerank_error:
                raise RerankError(index.rerank_error)
    finally:
        index.config = config
    scores = [score(name, cases, ranking, k) for name, ranking in rankings.items()]
    for threshold in sorted({*THRESHOLDS, config.rerank_threshold} - {0}):
        kept = [[r for r in ranking if r.relevance >= threshold / 100] for ranking in rankings["+ rerank"]]
        scores.append(score(f"+ threshold {threshold}%", cases, kept, k))
    return scores
