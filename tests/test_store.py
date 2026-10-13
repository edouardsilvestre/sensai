import pytest

from sensai.core.rag.models import Chunk
from sensai.core.rag.store import DocumentRecord, IndexMismatchError, StoreError, VectorStore
from tests.conftest import FakeClient


def record(path: str, sha: str = "sha") -> DocumentRecord:
    return DocumentRecord(
        path=path, name=path.rsplit("/", 1)[-1], kind="text", title="", metadata={"author": "Alice"},
        sha256=sha, size=1, mtime=1.0,
    )


def save(store: VectorStore, path: str, texts: list[str], model: str = "m") -> None:
    chunks = [Chunk(text, f"p. {i + 1}") for i, text in enumerate(texts)]
    store.save_document(record(path), chunks, [FakeClient.vector(text) for text in texts], model)


@pytest.fixture
def store(tmp_path):
    store = VectorStore(tmp_path / "index.db")
    yield store
    store.close()


def test_search_returns_closest_chunks_first(store):
    save(store, "/docs/a.txt", ["bank transfer to delta holding", "weather is sunny today"])
    save(store, "/docs/b.txt", ["minutes of the board meeting"])

    results = store.search(FakeClient.vector("transfer to delta"), k=2)
    assert [result.text for result in results][0] == "bank transfer to delta holding"
    assert results[0].source == "a.txt"
    assert results[0].location == "p. 1"
    assert results[0].description == "Author: Alice"
    assert results[0].score > results[1].score


def test_empty_index(store):
    assert store.is_empty()
    assert store.search(FakeClient.vector("anything"), k=3) == []


def test_saving_again_replaces_the_chunks(store):
    save(store, "/docs/a.txt", ["old text one", "old text two"])
    save(store, "/docs/a.txt", ["new text"])

    assert store.stats().documents == 1
    assert store.stats().chunks == 1
    assert [result.text for result in store.search(FakeClient.vector("old text"), k=5)] == ["new text"]


def test_deleting_a_document_removes_its_vectors(store):
    save(store, "/docs/a.txt", ["alpha"])
    save(store, "/docs/b.txt", ["beta"])
    store.delete_document(store.get_document("/docs/a.txt").id)

    assert [result.source for result in store.search(FakeClient.vector("alpha"), k=5)] == ["b.txt"]


def test_documents_under_matches_the_folder_only(store):
    save(store, "/docs/a.txt", ["a"])
    save(store, "/docs/sub/b.txt", ["b"])
    save(store, "/docs-other/c.txt", ["c"])
    assert sorted(document.name for document in store.documents_under("/docs")) == ["a.txt", "b.txt"]


def test_index_is_bound_to_its_embedding_model(store):
    save(store, "/docs/a.txt", ["alpha"], model="model-a")
    with pytest.raises(IndexMismatchError):
        store.check_model("model-b")
    with pytest.raises(IndexMismatchError):
        save(store, "/docs/b.txt", ["beta"], model="model-b")


def test_vectors_must_match_the_index_dimensions(store):
    save(store, "/docs/a.txt", ["alpha"])
    with pytest.raises(IndexMismatchError):
        store.search([0.1, 0.2], k=1)


def test_reset_allows_another_model(store):
    save(store, "/docs/a.txt", ["alpha"], model="model-a")
    store.reset()
    save(store, "/docs/a.txt", ["alpha"], model="model-b")
    assert store.embed_model == "model-b"
    assert store.stats().chunks == 1


def test_unusable_path_is_a_store_error(tmp_path):
    (tmp_path / "file").write_text("", encoding="utf-8")
    with pytest.raises(StoreError):
        VectorStore(tmp_path / "file" / "index.db")


def test_non_index_file_is_a_store_error(tmp_path):
    path = tmp_path / "notes.db"
    path.write_text("not a database" * 100, encoding="utf-8")
    with pytest.raises(StoreError):
        VectorStore(path)


def test_index_survives_reopening(tmp_path):
    first = VectorStore(tmp_path / "index.db")
    save(first, "/docs/a.txt", ["alpha"])
    first.close()

    second = VectorStore(tmp_path / "index.db")
    assert second.search(FakeClient.vector("alpha"), k=1)[0].text == "alpha"
    second.close()
