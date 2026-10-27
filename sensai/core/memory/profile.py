import os
import re
from pathlib import Path

PROFILE_HEADER = "About the user (standing preferences and instructions, follow them without being reminded):"

_BULLET = re.compile(r"^[-*]\s+")


class ProfileError(Exception):
    pass


class Profile:
    """
    The user's standing preferences and instructions, one per line of a text file that can also be edited by hand.
    It is read once at startup and sent to the model at the start of every conversation.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries = self._read()

    def _read(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ProfileError(f"Cannot read the profile {self.path}: {e}") from None
        entries = (_BULLET.sub("", line.strip()) for line in text.splitlines())
        return [entry for entry in entries if entry]

    def text(self) -> str:
        return "\n".join(f"- {entry}" for entry in self.entries)

    def add(self, entry: str) -> None:
        entry = " ".join(entry.split())
        if not entry:
            raise ProfileError("The instruction is empty")
        self.entries.append(entry)
        self._save()

    def remove(self, number: int) -> str:
        if not 1 <= number <= len(self.entries):
            raise ProfileError(f"No instruction {number}: the profile has {len(self.entries)}")
        entry = self.entries.pop(number - 1)
        self._save()
        return entry

    def _save(self) -> None:
        # Written next to the profile then renamed, so that an interrupted write never leaves half a profile.
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(self.text() + "\n" if self.entries else "", encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError as e:
            raise ProfileError(f"Cannot save the profile {self.path}: {e}") from None
