import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from sensai.core.config import Config
from sensai.core.llm.ollama import OllamaClient
from sensai.core.rag.chunker import Chunker, join_chunks, join_locations
from sensai.core.rag.embedder import Embedder
from sensai.core.rag.loaders import DocumentLoadError, is_supported, load_document
from sensai.core.rag.models import SearchFilter, SearchReport, SearchResult
from sensai.core.rag.reranker import RerankError, Reranker
from sensai.core.rag.retriever import fuse
from sensai.core.rag.store import DocumentRecord, IndexStats, VectorStore


@dataclass
class IngestProgress:
    position: int
    total: int
    name: str
    status: str
    chunks: int = 0
    error: str = ""


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    unsupported: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class _Passage:
    """The chunks `first` to `last` of a document, shown as one excerpt."""

    excerpt: SearchResult
    # None when the document of the excerpt is no longer in the store.
    document_id: int | None
    first: int
    last: int


class DocumentIndex:
    def __init__(self, store: VectorStore, embedder: Embedder, chunker: Chunker, reranker: Reranker, config: Config):
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.reranker = reranker
        # Why the reranker could not run: the searches go on without it, and do not load it again.
        self.rerank_error = ""
        # Read at each search, so that /feature hybrid off or /feature rerank off takes effect at once.
        self.config = config

    @classmethod
    def open(cls, config: Config, client: OllamaClient, rebuild: bool = False) -> "DocumentIndex":
        store = VectorStore(config.index_path, rebuild=rebuild)
        try:
            store.check_model(config.embed_model)
        except Exception:
            store.close()
            raise
        embedder = Embedder(client, config.embed_model, max_chars=config.chunk_size)
        chunker = Chunker(config.chunk_size, config.chunk_overlap)
        return cls(store, embedder, chunker, Reranker(config.rerank_model), config)

    def ingest(self, paths: list[Path], on_progress: Callable[[IngestProgress], None] | None = None) -> IngestReport:
        started = time.monotonic()
        report = IngestReport()
        files: dict[Path, str] = {}
        for root in paths:
            root = root.expanduser().resolve()
            if not root.exists():
                report.failed.append((str(root), "not found"))
            elif root.is_file():
                if is_supported(root):
                    files.setdefault(root, root.name)
                else:
                    report.failed.append((root.name, f"unsupported file type '{root.suffix or root.name}'"))
            else:
                for path in _visible_files(root):
                    if is_supported(path):
                        files.setdefault(path, f"{root.name}/{path.relative_to(root).as_posix()}")
                    else:
                        report.unsupported += 1
                report.removed += self._prune(root)

        for position, (path, name) in enumerate(files.items(), start=1):
            progress = self._ingest_file(path, name)
            progress.position, progress.total = position, len(files)
            if progress.status == "failed":
                report.failed.append((name, progress.error))
            else:
                setattr(report, progress.status, getattr(report, progress.status) + 1)
                report.chunks += progress.chunks
            if on_progress:
                on_progress(progress)
        report.seconds = time.monotonic() - started
        return report

    def _ingest_file(self, path: Path, name: str) -> IngestProgress:
        try:
            stat = path.stat()
            stored = self.store.get_document(str(path))
            if stored and stored.size == stat.st_size and stored.mtime == stat.st_mtime:
                return IngestProgress(0, 0, name, "unchanged")
            digest = _sha256(path)
            if stored and stored.sha256 == digest:
                self.store.update_file_info(stored.id, stat.st_size, stat.st_mtime)
                return IngestProgress(0, 0, name, "unchanged")
            document = load_document(path)
        except (DocumentLoadError, OSError) as e:
            return IngestProgress(0, 0, name, "failed", error=str(e))

        chunks = self.chunker.split(document)
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        record = DocumentRecord(
            path=str(path),
            name=name,
            kind=document.kind,
            title=document.title,
            metadata=document.metadata,
            sha256=digest,
            size=stat.st_size,
            mtime=stat.st_mtime,
        )
        self.store.save_document(record, chunks, vectors, self.embedder.model)
        return IngestProgress(0, 0, name, "updated" if stored else "added", chunks=len(chunks))

    def _prune(self, folder: Path) -> int:
        removed = 0
        for document in self.store.documents_under(str(folder)):
            if not Path(document.path).exists():
                self.store.delete_document(document.id)
                removed += 1
        return removed

    def search(
        self, query: str, k: int, filters: Sequence[SearchFilter] = (), report: SearchReport | None = None
    ) -> list[SearchResult]:
        if self.store.is_empty():
            return []
        document_ids = self.store.allowed_documents(filters)
        if document_ids == []:
            return []
        vector = self.embedder.embed_query(query)
        candidates = max(self.config.rag_candidates, k)
        found = self.store.search(vector, candidates, document_ids)
        if self.config.features["hybrid"]:
            keyword = self.store.keyword_search(query, candidates, vector, document_ids)
            found = fuse([found, keyword])[:candidates]
        if self.config.features["rerank"] and not self.rerank_error:
            try:
                found = self.reranker.rerank(query, found)
            except RerankError as e:
                self.rerank_error = str(e)
            else:
                threshold = self.config.rerank_threshold / 100
                if report is not None:
                    report.candidates = len(found)
                    report.dropped = [result for result in found if result.relevance < threshold]
                found = [result for result in found if result.relevance >= threshold]
        if self.config.features["rerank"] and self.rerank_error and report is not None:
            report.rerank_error = self.rerank_error
        return self._widen(found[:k])

    def _widen(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Each excerpt with the chunks around it, to read the whole passage while searching small chunks.
        The passages that touch in a document make one, under the most relevant of their excerpts.
        """
        around = self.config.rag_neighbors
        if not around or not results:
            return results
        positions = self.store.chunk_positions([result.chunk_id for result in results])
        passages: list[_Passage] = []
        for result in results:
            if result.chunk_id not in positions:
                passages.append(_Passage(result, None, 0, -1))
                continue
            document_id, position = positions[result.chunk_id]
            new = _Passage(result, document_id, max(position - around, 0), position + around)
            touching = [
                p
                for p in passages
                if p.document_id == document_id and p.first <= new.last + 1 and new.first <= p.last + 1
            ]
            if not touching:
                passages.append(new)
                continue
            kept, merged = touching[0], touching[1:]
            kept.first = min(p.first for p in touching + [new])
            kept.last = max(p.last for p in touching + [new])
            passages = [p for p in passages if p not in merged]
        return [self._read(passage) for passage in passages]

    def _read(self, passage: _Passage) -> SearchResult:
        """The excerpt of a passage with the text of all its chunks, or alone when they are no longer in the store."""
        if passage.document_id is None:
            return passage.excerpt
        chunks = self.store.chunks_between(passage.document_id, passage.first, passage.last)
        if not chunks:
            return passage.excerpt
        location = join_locations([chunk.location for chunk in chunks])
        return replace(passage.excerpt, text=join_chunks(chunks), location=location)

    def count_documents(self, filters: Sequence[SearchFilter]) -> int:
        document_ids = self.store.allowed_documents(filters)
        return self.stats().documents if document_ids is None else len(document_ids)

    def stats(self) -> IndexStats:
        return self.store.stats()

    def fingerprint(self) -> str:
        return self.store.fingerprint()

    def close(self) -> None:
        self.store.close()


def _visible_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
