import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

OUTCOMES = ("answered", "cached", "blocked", "stopped", "error")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    persona TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    question TEXT NOT NULL,
    prompt TEXT NOT NULL,
    answer TEXT NOT NULL,
    outcome TEXT NOT NULL,
    error TEXT NOT NULL,
    seconds REAL NOT NULL,
    first_token_seconds REAL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    tools TEXT NOT NULL,
    sources INTEGER NOT NULL,
    guardrails TEXT NOT NULL,
    reasoning_steps INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS interactions_created_at ON interactions(created_at);
"""

_COLUMNS = (
    "created_at", "session_id", "model", "persona", "prompt_version", "question", "prompt", "answer", "outcome",
    "error", "seconds", "first_token_seconds", "prompt_tokens", "completion_tokens", "tools", "sources",
    "guardrails", "reasoning_steps",
)
_JSON_COLUMNS = {"prompt", "tools", "guardrails"}


class LogError(Exception):
    pass


@dataclass
class Interaction:
    model: str
    question: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    persona: str = ""
    prompt_version: str = ""
    prompt: list[dict] = field(default_factory=list)
    answer: str = ""
    outcome: str = "answered"
    error: str = ""
    seconds: float = 0.0
    first_token_seconds: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tools: list[str] = field(default_factory=list)
    sources: int = 0
    guardrails: list[str] = field(default_factory=list)
    reasoning_steps: int = 0


def _value(interaction: Interaction, column: str) -> object:
    value = getattr(interaction, column)
    if column == "created_at":
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    if column in _JSON_COLUMNS:
        return json.dumps(value, ensure_ascii=False)
    return value


def _interaction(row: tuple) -> Interaction:
    values = dict(zip(_COLUMNS, row))
    for column in _JSON_COLUMNS:
        values[column] = json.loads(values[column])
    values["created_at"] = datetime.fromisoformat(values["created_at"])
    return Interaction(**values)


class InteractionLog:
    def __init__(self, path: Path):
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path)
        except (OSError, sqlite3.Error) as e:
            raise LogError(f"Cannot open the log at {path}: {e}") from e
        try:
            self._db.executescript(_SCHEMA)
        except sqlite3.DatabaseError as e:
            self._db.close()
            raise LogError(f"Cannot open the log at {path}: {e}") from e

    def add(self, interaction: Interaction) -> None:
        if interaction.outcome not in OUTCOMES:
            raise LogError(f"Unknown outcome '{interaction.outcome}': expected one of {', '.join(OUTCOMES)}")
        placeholders = ", ".join("?" for _ in _COLUMNS)
        try:
            with self._db:
                self._db.execute(
                    f"INSERT INTO interactions ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                    [_value(interaction, column) for column in _COLUMNS],
                )
        except sqlite3.Error as e:
            raise LogError(f"Cannot write to the log at {self.path}: {e}") from e

    def since(self, start: datetime) -> list[Interaction]:
        try:
            rows = self._db.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM interactions WHERE created_at >= ? ORDER BY created_at, id",
                (start.astimezone(timezone.utc).isoformat(timespec="milliseconds"),),
            ).fetchall()
        except sqlite3.Error as e:
            raise LogError(f"Cannot read the log at {self.path}: {e}") from e
        return [_interaction(row) for row in rows]

    def clear(self) -> int:
        try:
            with self._db:
                return self._db.execute("DELETE FROM interactions").rowcount
        except sqlite3.Error as e:
            raise LogError(f"Cannot clear the log at {self.path}: {e}") from e

    def close(self) -> None:
        self._db.close()
