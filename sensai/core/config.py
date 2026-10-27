import os
from dataclasses import dataclass, field
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

    compact_threshold: int = 75
    keep_recent_turns: int = 4

    sessions_dir: Path = Path(".sensai/sessions")
    profile_path: Path = Path(".sensai/profile.md")

    embed_model: str = "qwen3-embedding:0.6b"
    index_path: Path = Path(".sensai/index.db")
    rag_top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200

    tools: bool = True
    web_search: bool = False
    ollama_api_key: str = field(default="", repr=False)
    web_search_results: int = 5
    max_tool_rounds: int = 5


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
        compact_threshold=_int_env("SENSAI_COMPACT_THRESHOLD", Config.compact_threshold, minimum=10, maximum=95),
        keep_recent_turns=_int_env("SENSAI_KEEP_RECENT_TURNS", Config.keep_recent_turns, minimum=1),
        sessions_dir=Path(os.getenv("SENSAI_SESSIONS_DIR", str(Config.sessions_dir))),
        profile_path=Path(os.getenv("SENSAI_PROFILE_PATH", str(Config.profile_path))),
        embed_model=os.getenv("SENSAI_EMBED_MODEL", Config.embed_model),
        index_path=Path(os.getenv("SENSAI_INDEX_PATH", str(Config.index_path))),
        rag_top_k=_int_env("SENSAI_RAG_TOP_K", Config.rag_top_k, minimum=1, maximum=50),
        chunk_size=_int_env("SENSAI_RAG_CHUNK_SIZE", Config.chunk_size, minimum=100),
        chunk_overlap=_int_env("SENSAI_RAG_CHUNK_OVERLAP", Config.chunk_overlap, minimum=0),
        tools=_bool_env("SENSAI_TOOLS") is not False,
        web_search=_bool_env("SENSAI_WEB_SEARCH") or False,
        ollama_api_key=os.getenv("OLLAMA_API_KEY", Config.ollama_api_key).strip(),
        web_search_results=_int_env("SENSAI_WEB_SEARCH_RESULTS", Config.web_search_results, minimum=1, maximum=10),
        max_tool_rounds=_int_env("SENSAI_MAX_TOOL_ROUNDS", Config.max_tool_rounds, minimum=1, maximum=20),
    )
    if config.chunk_overlap >= config.chunk_size:
        raise ConfigError("SENSAI_RAG_CHUNK_OVERLAP must be smaller than SENSAI_RAG_CHUNK_SIZE")
    if config.tools and config.web_search and not config.ollama_api_key:
        raise ConfigError("SENSAI_WEB_SEARCH needs OLLAMA_API_KEY: create a key on https://ollama.com/settings/keys")
    return config
