import json
import os
import re
import secrets
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sensai.core.config import Config

SHARED_PATH = "/shared"
TOKEN_BYTES = 16
# A link is a file name: nothing that could point outside the folder. Hexadecimal, as Windows ignores the case of a name.
_TOKEN = re.compile(rf"^[0-9a-f]{{{TOKEN_BYTES * 2}}}$")


class ShareError(Exception):
    pass


@dataclass
class Share:
    token: str
    kind: str
    title: str
    created: datetime


class ShareStore:
    """
    One JSON file per public page, named after its link. The page is a frozen copy: nothing in it follows the
    conversation, and removing the file takes the link down at once, without restarting the server.
    """

    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, token: str) -> Path | None:
        return self.directory / f"{token}.json" if _TOKEN.match(token) else None

    def add(self, kind: str, title: str, page: str) -> Share:
        share = Share(secrets.token_hex(TOKEN_BYTES), kind, title, datetime.now())
        data = {"kind": kind, "title": title, "created": share.created.isoformat(timespec="seconds"), "page": page}
        path = self.directory / f"{share.token}.json"
        # Written next to the page then renamed, so that a link never shows half a page.
        temporary = path.with_name(path.name + ".tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as e:
            raise ShareError(f"Cannot write the page to {path}: {e}") from None
        return share

    def _read(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def page(self, token: str) -> str | None:
        """The page of a link, None when the link does not exist or was taken down."""
        path = self._path(token)
        data = self._read(path) if path else None
        return data["page"] if data and isinstance(data.get("page"), str) else None

    def list(self) -> list[Share]:
        """The public links, the most recent first. Unreadable files are left out."""
        shares = []
        for path in self.directory.glob("*.json") if self.directory.is_dir() else []:
            data = self._read(path) if self._path(path.stem) else None
            try:
                shares.append(Share(path.stem, str(data["kind"]), str(data["title"]), datetime.fromisoformat(data["created"])))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(shares, key=lambda share: (share.created, share.token), reverse=True)

    def remove(self, token: str) -> bool:
        path = self._path(token)
        if path is None:
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as e:
            raise ShareError(f"Cannot take the link down: {e}") from None
        return True


def local_address() -> str:
    """The address of this machine on its network. Nothing is sent: a UDP socket only picks the way out when it connects."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def public_base(config: Config) -> str:
    return config.public_url.rstrip("/") or f"http://{local_address()}:{config.public_port}"


def public_link(config: Config, share: Share) -> str:
    return f"{public_base(config)}{SHARED_PATH}/{share.token}"
