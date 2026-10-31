import json
import math
import sqlite3
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from sensai.core.rag.models import SearchResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY,
    context TEXT NOT NULL,
    question TEXT NOT NULL,
    vector BLOB NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,
    created_at TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS answers_context ON answers(context);
"""


class CacheError(Exception):
    pass


@dataclass
class CachedAnswer:
    question: str
    answer: str
    sources: list[SearchResult]
    created_at: datetime
    similarity: float


@dataclass
class CacheStats:
    answers: int
    uses: int


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


class SemanticCache:
    def __init__(self, path: Path, embed: Callable[[str], list[float]], threshold: float):
        self.path = path
        self.embed = embed
        self.threshold = threshold
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path)
        except (OSError, sqlite3.Error) as e:
            raise CacheError(f"Cannot open the cache at {path}: {e}") from e
        try:
            self._db.executescript(_SCHEMA)
        except sqlite3.DatabaseError as e:
            self._db.close()
            raise CacheError(f"Cannot open the cache at {path}: {e}") from e

    def find(self, vector: list[float], context: str) -> CachedAnswer | None:
        rows = self._db.execute(
            "SELECT id, question, vector, answer, sources, created_at FROM answers WHERE context = ?", (context,)
        ).fetchall()
        scored = [(_cosine(vector, array("f", row[2]).tolist()), row) for row in rows]
        similarity, row = max(scored, key=lambda pair: pair[0], default=(0.0, None))
        if row is None or similarity < self.threshold:
            return None
        entry_id, question, _, answer, sources, created_at = row
        with self._db:
            self._db.execute("UPDATE answers SET uses = uses + 1 WHERE id = ?", (entry_id,))
        results = [SearchResult(**source) for source in json.loads(sources)]
        return CachedAnswer(question, answer, results, datetime.fromisoformat(created_at), similarity)

    def add(self, question: str, vector: list[float], context: str, answer: str, sources: list[SearchResult]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO answers (context, question, vector, answer, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    context, question, array("f", vector).tobytes(), answer,
                    json.dumps([asdict(source) for source in sources], ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def stats(self) -> CacheStats:
        answers, uses = self._db.execute("SELECT COUNT(*), COALESCE(SUM(uses), 0) FROM answers").fetchone()
        return CacheStats(answers, uses)

    def clear(self) -> int:
        with self._db:
            return self._db.execute("DELETE FROM answers").rowcount

    def close(self) -> None:
        self._db.close()
