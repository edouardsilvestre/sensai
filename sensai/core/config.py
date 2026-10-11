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

