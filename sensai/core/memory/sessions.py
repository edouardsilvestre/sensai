import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from sensai.core.document.document import Document
from sensai.core.memory.conversation import Branch, ConversationMemory, Turn

FORMAT_VERSION = 1
TITLE_CHARS = 60
# The conversations of the scheduled tasks: they are listed, but are never "the last conversation".
TASK_PREFIX = "task-"

# Ids are file names: nothing that could point outside the sessions folder.
_ID = re.compile(r"^[\w-]+$")


class SessionError(Exception):
    pass


@dataclass
class SessionInfo:
    id: str
    updated: datetime
    turns: int
    title: str


def _title(turns: list[Turn]) -> str:
    text = " ".join(turns[0].user.split()) if turns else "(condensed conversation)"
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 3].rstrip() + "..."


def _position(branch: Branch) -> dict:
    return {"number": branch.number, "parent": branch.parent, "turn": branch.turn, "label": branch.label}


def _saved_branch(branch: Branch) -> dict:
    return _position(branch) | {
        "summary": branch.summary, "turns": [asdict(turn) for turn in branch.turns], "document": branch.document.markdown(),
    }


def _read_turns(saved: list[dict]) -> list[Turn]:
    return [
        Turn(turn["user"], turn["assistant"], list(turn.get("citations", [])), str(turn.get("document", "")))
        for turn in saved
    ]


def _read_branch(saved: dict) -> Branch:
    parent = saved["parent"]
    return Branch(
        int(saved["number"]), None if parent is None else int(parent), int(saved["turn"]), str(saved["label"]),
        str(saved.get("summary", "")), _read_turns(saved.get("turns", [])), Document(saved.get("document", "")),
    )


class SessionStore:
    """
    One JSON file per conversation, holding the summary and the turns kept verbatim, as the memory has them.
    The top level is the active branch, and `other_branches` the others.
    """

    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, session_id: str) -> Path:
        if not _ID.match(session_id):
            raise SessionError(f"Invalid session id '{session_id}'")
        return self.directory / f"{session_id}.json"

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def new_id(self) -> str:
        base = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_id, number = base, 2
        while self._path(session_id).exists():
            session_id, number = f"{base}-{number}", number + 1
        return session_id

    def save(
        self, session_id: str, memory: ConversationMemory, model: str,
        persona: str | None = None, prompt_version: int | None = None,
    ) -> None:
        data = {
            "version": FORMAT_VERSION,
            "id": session_id,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            # Which system prompt the last answers came from: /prompts lists the versions.
            "persona": persona,
            "prompt_version": prompt_version,
            "summary": memory.summary,
            "turns": [asdict(turn) for turn in memory.turns],
            "document": memory.document.markdown(),
            "branch": _position(memory.branch),
            "other_branches": [_saved_branch(branch) for branch in memory.other_branches],
        }
        path = self._path(session_id)
        # Written next to the session then renamed, so that a crash never leaves half a conversation.
        temporary = path.with_name(path.name + ".tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as e:
            raise SessionError(f"Cannot save the conversation to {path}: {e}") from None

    def _read(self, path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            turns = _read_turns(data["turns"])
            # Sessions saved before the document existed start with an empty one.
            document = Document(data["document"]) if isinstance(data.get("document"), str) else Document()
            return {
                "summary": str(data.get("summary", "")), "turns": turns, "document": document,
                "updated": datetime.fromisoformat(data["updated"]),
                # Sessions saved before the branches have only one.
                "branch": _read_branch(data["branch"]) if "branch" in data else Branch(),
                "other_branches": [_read_branch(branch) for branch in data.get("other_branches", [])],
            }
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
            raise SessionError(f"Cannot read the conversation {path.name}: {e}") from None

    def load(self, session_id: str, memory: ConversationMemory) -> int:
        """Replaces the memory's history with the saved one. Returns the number of turns restored."""
        path = self._path(session_id)
        if not path.exists():
            raise SessionError(f"No saved conversation '{session_id}': /sessions lists them")
        data = self._read(path)
        memory.summary, memory.turns, memory.last_usage = data["summary"], data["turns"], None
        memory.document = data["document"]
        memory.branch, memory.other_branches = data["branch"], data["other_branches"]
        return len(data["turns"])

    def list(self) -> list[SessionInfo]:
        """The saved conversations, the most recent first. Unreadable files are left out."""
        sessions = []
        for path in self.directory.glob("*.json") if self.directory.is_dir() else []:
            try:
                data = self._read(path)
            except SessionError:
                continue
            sessions.append(SessionInfo(path.stem, data["updated"], len(data["turns"]), _title(data["turns"])))
        return sorted(sessions, key=lambda session: (session.updated, session.id), reverse=True)

    def latest(self) -> str | None:
        return next((session.id for session in self.list() if not session.id.startswith(TASK_PREFIX)), None)
