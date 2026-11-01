from sensai.core.rag.models import SearchResult
from sensai.core.rag.retriever import fuse


def result(chunk_id: int, score: float = 0.5) -> SearchResult:
    return SearchResult(chunk_id, f"chunk {chunk_id}", "", "doc.txt", "", score)


def test_a_chunk_found_by_both_searches_comes_first():
    a, b, c = result(1), result(2), result(3)
    semantic = [a, result(9), b]
    keyword = [c, b]
    assert [r.chunk_id for r in fuse([semantic, keyword])] == [2, 1, 3, 9]


def test_the_fusion_keeps_the_first_copy_of_each_chunk():
    semantic = [result(1, score=0.8)]
    keyword = [result(1, score=0.0)]
    [fused] = fuse([semantic, keyword])
    assert fused.score == 0.8


def test_a_single_ranking_keeps_its_order():
    assert [r.chunk_id for r in fuse([[result(3), result(1), result(2)]])] == [3, 1, 2]
    assert fuse([[], []]) == []
