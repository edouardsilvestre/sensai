import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sensai.core.config import Config
from sensai.core.llm.ollama import OllamaClient
from sensai.core.rag.chunker import Chunker
from sensai.core.rag.embedder import Embedder
from sensai.core.rag.loaders import DocumentLoadError, is_supported, load_document
from sensai.core.rag.models import SearchResult
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


class DocumentIndex:
    def __init__(self, store: VectorStore, embedder: Embedder, chunker: Chunker):
        self.store = store
        self.embedder = embedder
        self.chunker = chunker

    @classmethod
    def open(cls, config: Config, client: OllamaClient, rebuild: bool = False) -> "DocumentIndex":
        store = VectorStore(config.index_path)
        try:
            if rebuild:
                store.reset()
            store.check_model(config.embed_model)
        except Exception:
            store.close()
            raise
        embedder = Embedder(client, config.embed_model, max_chars=config.chunk_size)
        return cls(store, embedder, Chunker(config.chunk_size, config.chunk_overlap))

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

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self.store.is_empty():
            return []
        return self.store.search(self.embedder.embed_query(query), k)

    def stats(self) -> IndexStats:
        return self.store.stats()

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
