import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ENV_PATH = Path(".env")


class ConfigError(Exception):
    pass


@dataclass
class Config:
    model: str = "llama3"
    host: str = "http://localhost:11434"
    system_prompt: str = ""
    num_ctx: int = 8192
    think: bool | None = None

    embed_model: str = "qwen3-embedding:0.6b"
    index_path: Path = Path(".sensai/index.db")
    rag_top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200


def _int_env(name: str, default: int, minimum: int, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got '{raw}'") from None
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ConfigError(f"{name} must be {bounds}, got {value}")
    return value


def _bool_env(name: str) -> bool | None:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false, got '{raw}'")


def load_config(path: Path | None = None) -> Config:
    file_path = path or DEFAULT_ENV_PATH
    if path is not None and not file_path.exists():
        raise ConfigError(f"Env file not found: {file_path}")
    load_dotenv(file_path)

    config = Config(
        model=os.getenv("SENSAI_MODEL", Config.model),
        host=os.getenv("OLLAMA_HOST", Config.host),
        system_prompt=os.getenv("SENSAI_SYSTEM_PROMPT", Config.system_prompt),
        num_ctx=_int_env("SENSAI_NUM_CTX", Config.num_ctx, minimum=512),
        think=_bool_env("SENSAI_THINK"),
        embed_model=os.getenv("SENSAI_EMBED_MODEL", Config.embed_model),
        index_path=Path(os.getenv("SENSAI_INDEX_PATH", str(Config.index_path))),
        rag_top_k=_int_env("SENSAI_RAG_TOP_K", Config.rag_top_k, minimum=1, maximum=50),
        chunk_size=_int_env("SENSAI_RAG_CHUNK_SIZE", Config.chunk_size, minimum=100),
        chunk_overlap=_int_env("SENSAI_RAG_CHUNK_OVERLAP", Config.chunk_overlap, minimum=0),
    )
    if config.chunk_overlap >= config.chunk_size:
        raise ConfigError("SENSAI_RAG_CHUNK_OVERLAP must be smaller than SENSAI_RAG_CHUNK_SIZE")
    return config
