import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ENV_PATH = Path(".env")
FEATURE_PREFIX = "SENSAI_FEATURE_"
API_KEY_HELP = "create a key on https://ollama.com/settings/keys"
# Settings that became features: an old .env must not be silently ignored, SENSAI_TOOLS=false would turn tools on.
RENAMED = {"SENSAI_TOOLS": "SENSAI_FEATURE_TOOLS", "SENSAI_WEB_SEARCH": "SENSAI_FEATURE_WEB_SEARCH"}


@dataclass(frozen=True)
class Feature:
    description: str
    default: bool = True


FEATURES = {
    "rag": Feature("answer from the indexed documents"),
    "tools": Feature("let the model call the tools enabled in the configuration"),
    # Off by default: the queries leave the machine.
    "web_search": Feature("let the model search the web; the queries are sent to Ollama's web search", default=False),
}


class ConfigError(Exception):
    pass


def default_features() -> dict[str, bool]:
    return {name: feature.default for name, feature in FEATURES.items()}


@dataclass
class Config:
    model: str = "llama3"
    host: str = "http://localhost:11434"
    system_prompt: str = ""
    num_ctx: int = 8192
    think: bool | None = None
    features: dict[str, bool] = field(default_factory=default_features)

    compact_threshold: int = 75
    keep_recent_turns: int = 4

    embed_model: str = "qwen3-embedding:0.6b"
    index_path: Path = Path(".sensai/index.db")
    rag_top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200

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


def _features_env() -> dict[str, bool]:
    for old, new in RENAMED.items():
        if os.getenv(old) is not None:
            raise ConfigError(f"{old} is now {new}: rename it in your .env")
    names = [name.removeprefix(FEATURE_PREFIX).lower() for name in os.environ if name.startswith(FEATURE_PREFIX)]
    unknown = sorted(FEATURE_PREFIX + name.upper() for name in names if name not in FEATURES)
    if unknown:
        known = ", ".join(FEATURE_PREFIX + name.upper() for name in FEATURES)
        raise ConfigError(f"Unknown feature {', '.join(unknown)}. Known features: {known}")
    features = default_features()
    for name in FEATURES:
        value = _bool_env(FEATURE_PREFIX + name.upper())
        if value is not None:
            features[name] = value
    return features


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
        features=_features_env(),
        compact_threshold=_int_env("SENSAI_COMPACT_THRESHOLD", Config.compact_threshold, minimum=10, maximum=95),
        keep_recent_turns=_int_env("SENSAI_KEEP_RECENT_TURNS", Config.keep_recent_turns, minimum=1),
        embed_model=os.getenv("SENSAI_EMBED_MODEL", Config.embed_model),
        index_path=Path(os.getenv("SENSAI_INDEX_PATH", str(Config.index_path))),
        rag_top_k=_int_env("SENSAI_RAG_TOP_K", Config.rag_top_k, minimum=1, maximum=50),
        chunk_size=_int_env("SENSAI_RAG_CHUNK_SIZE", Config.chunk_size, minimum=100),
        chunk_overlap=_int_env("SENSAI_RAG_CHUNK_OVERLAP", Config.chunk_overlap, minimum=0),
        ollama_api_key=os.getenv("OLLAMA_API_KEY", Config.ollama_api_key).strip(),
        web_search_results=_int_env("SENSAI_WEB_SEARCH_RESULTS", Config.web_search_results, minimum=1, maximum=10),
        max_tool_rounds=_int_env("SENSAI_MAX_TOOL_ROUNDS", Config.max_tool_rounds, minimum=1, maximum=20),
    )
    if config.chunk_overlap >= config.chunk_size:
        raise ConfigError("SENSAI_RAG_CHUNK_OVERLAP must be smaller than SENSAI_RAG_CHUNK_SIZE")
    if config.features["tools"] and config.features["web_search"] and not config.ollama_api_key:
        raise ConfigError(f"SENSAI_FEATURE_WEB_SEARCH needs OLLAMA_API_KEY: {API_KEY_HELP}")
    return config
