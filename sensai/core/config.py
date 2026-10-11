"""Configuration loading: defaults < .env file < environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ENV_PATH = Path(".env")


class ConfigError(Exception):
    """Raised when the configuration file is missing."""


@dataclass
class Config:
    model: str = "llama3"
    host: str = "http://localhost:11434"
    system_prompt: str = ""


def load_config(path: Path | None = None) -> Config:
    """Build a Config from the environment, loaded from an optional .env file.

    An explicit `path` must exist; the default .env is optional.
    """
    file_path = path or DEFAULT_ENV_PATH
    if path is not None and not file_path.exists():
        raise ConfigError(f"Env file not found: {file_path}")
    load_dotenv(file_path)

    return Config(
        model=os.getenv("SENSAI_MODEL", Config.model),
        host=os.getenv("OLLAMA_HOST", Config.host),
        system_prompt=os.getenv("SENSAI_SYSTEM_PROMPT", Config.system_prompt),
    )
