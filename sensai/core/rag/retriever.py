from sensai.core.rag.models import SearchResult

# The usual constant of Reciprocal Rank Fusion: the lower ranks still count, without one first rank taking it all.
RRF_K = 60


def fuse(rankings: list[list[SearchResult]], k: int = RRF_K) -> list[SearchResult]:
    """
    Merges rankings whose scores cannot be compared, like a cosine similarity and BM25, by their ranks only:
    each chunk scores the sum of 1 / (k + rank) over the rankings it is in.
    """
    scores: dict[int, float] = {}
    results: dict[int, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (k + rank)
            results.setdefault(result.chunk_id, result)
    return sorted(results.values(), key=lambda result: scores[result.chunk_id], reverse=True)
