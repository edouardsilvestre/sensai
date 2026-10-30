from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

# Never readable nor writable, whatever the folders allowed: the settings with the API key, Sensai's own data
# (the index, the sessions, the memory and the guardrail rules, which hold the names to protect) and git.
ALWAYS_DENIED = (".env", ".env.*", ".sensai", ".git")


class PermissionDenied(Exception):
    pass


@dataclass
class Permissions:
    """
    The folders the model may read and the ones it may write, checked before every file operation.
    A path is resolved first, following `..` and links, so that it cannot leave an allowed folder by a detour.
    """

    read: list[Path] = field(default_factory=list)
    write: list[Path] = field(default_factory=list)
    # Patterns matched against each part of the path inside its folder, like "sources" or "*.key".
    deny: list[str] = field(default_factory=list)
    base: Path = field(default_factory=Path.cwd)

    def __post_init__(self):
        self.read = [self._resolve(path) for path in self.read]
        self.write = [self._resolve(path) for path in self.write]
        self.deny = [pattern.strip("/\\") for pattern in self.deny if pattern.strip("/\\")]

    def _resolve(self, path: Path | str) -> Path:
        path = Path(path).expanduser()
        return (path if path.is_absolute() else self.base / path).resolve()

    def _folder(self, path: Path, folders: list[Path]) -> Path | None:
        return next((folder for folder in folders if path == folder or path.is_relative_to(folder)), None)

    def _denied(self, path: Path, folder: Path) -> str | None:
        # The built-in patterns apply to the whole path, even when an allowed folder is inside one of them;
        # the user's apply inside the folder, so that "sources" does not deny a folder that sits under /home/sources.
        checks = [(path.parts[1:], ALWAYS_DENIED), (path.relative_to(folder).parts, self.deny)]
        for parts, patterns in checks:
            for part in parts:
                for pattern in patterns:
                    if fnmatch(part.casefold(), pattern.casefold()):
                        return pattern
        return None

    def check(self, path: str, write: bool = False) -> Path:
        """Returns the resolved path, or raises PermissionDenied with the reason. Writing allows reading."""
        if not str(path).strip():
            raise PermissionDenied("no path given")
        resolved = self._resolve(path)
        folders = self.write if write else self.write + self.read
        folder = self._folder(resolved, folders)
        if folder is None:
            action, setting = ("write", "SENSAI_FILES_WRITE") if write else ("read", "SENSAI_FILES_READ")
            allowed = ", ".join(self.display(folder) for folder in folders) or "none"
            raise PermissionDenied(f"{path} is outside the folders you may {action} ({setting}: {allowed})")
        pattern = self._denied(resolved, folder)
        if pattern is not None:
            raise PermissionDenied(f"{path} is denied by the pattern '{pattern}'")
        return resolved

    def allowed(self, path: Path) -> bool:
        """For listing a folder: true when the entry may be read, without saying why not."""
        try:
            self.check(str(path))
        except PermissionDenied:
            return False
        return True

    def display(self, path: Path) -> str:
        """The path relative to where Sensai runs when it is inside, so that the model can reuse it as is."""
        try:
            relative = path.relative_to(self.base.resolve())
        except ValueError:
            return str(path)
        return str(relative) if relative.parts else "."
