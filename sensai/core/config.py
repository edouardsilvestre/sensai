import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_ENV_PATH = Path(".env")
DEFAULT_PERSONAS_DIR = Path(__file__).resolve().parents[2] / "personas"
FEATURE_PREFIX = "SENSAI_FEATURE_"
API_KEY_HELP = "create a key on https://ollama.com/settings/keys"
# Settings that became features: an old .env must not be silently ignored, SENSAI_TOOLS=false would turn tools on.
RENAMED = {"SENSAI_TOOLS": "SENSAI_FEATURE_TOOLS", "SENSAI_WEB_SEARCH": "SENSAI_FEATURE_WEB_SEARCH"}


@dataclass(frozen=True)
class Feature:
    description: str
    default: bool = True
    # True when the feature enables a tool the model can call himself.
    tool: bool = False


FEATURES = {
    "rag": Feature("answer from the indexed documents"),
    "tools": Feature("let the model call the tools enabled in the configuration"),
    # Off by default: the queries leave the machine.
    "web_search": Feature("let the model search the web; the queries are sent to Ollama's web search", default=False, tool=True),
    # Off by default: the code runs on this machine, and needs Docker.
    "code_exec": Feature("let the model run Python code in a Docker sandbox", default=False, tool=True),
    # Off by default: the model reads and writes files on this machine, within SENSAI_FILES_READ and SENSAI_FILES_WRITE.
    "files": Feature("let the model read and write files in the folders you allow", default=False, tool=True),
    "memory": Feature("let the model save and look up facts about people, companies and places", tool=True),
    # Without a file at mcp_path, no server is started.
    "mcp": Feature("let the model call the tools of the MCP servers listed in the MCP file", tool=True),
    "questions": Feature("let the model ask you a question when your request is ambiguous", tool=True),
    "sessions": Feature("save every conversation so that it can be resumed"),
    "profile": Feature("send your profile to the model at the start of every conversation"),
    "document": Feature("keep a working document of the established facts, updated after every answer"),
    "guardrails": Feature("check the messages, the answers and the tool calls against the guardrail rules"),
    "orchestration": Feature("let /review have the answer checked by the fact-checker and written by the editor"),
    "branching": Feature("edit a past message into a new branch, and keep an interrupted answer to redirect it"),
    "reasoning": Feature("reason step by step and show each step before answering", default=False),
    "cache": Feature("reuse the answer to a close question already asked, once you confirm it"),
    "examples": Feature("show the model your answers kept as examples, for similar questions"),
    "prompt_versions": Feature("keep every version of the system prompt, to compare them and roll back"),
    "schedule": Feature("run the scheduled tasks and write their reports"),
    "logging": Feature("log every message with its prompt, answer, tools, time and tokens"),
    "fact_check": Feature("check each answer against its excerpts and flag what they do not support", default=False),
}
APPROVAL_MODES = ("critical", "all", "off")


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
    persona: str = ""
    personas_dir: Path = DEFAULT_PERSONAS_DIR
    features: dict[str, bool] = field(default_factory=default_features)

    compact_threshold: int = 75
    keep_recent_turns: int = 4

    sessions_dir: Path = Path(".sensai/sessions")
    profile_path: Path = Path(".sensai/profile.md")
    memory_path: Path = Path(".sensai/memory.db")
    guardrails_path: Path = Path(".sensai/guardrails.json")
    mcp_path: Path = Path(".sensai/mcp.json")
    examples_path: Path = Path(".sensai/examples.json")
    examples_k: int = 2
    prompts_dir: Path = Path(".sensai/prompts")
    prompt_cases_path: Path = Path(".sensai/prompt_cases.json")
    schedule_path: Path = Path(".sensai/schedule.json")
    reports_dir: Path = Path(".sensai/reports")
    judge_model: str = ""
    judge_max_tokens: int = 1500

    embed_model: str = "qwen3-embedding:0.6b"
    index_path: Path = Path(".sensai/index.db")
    rag_top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200
    cache_path: Path = Path(".sensai/cache.db")
    cache_threshold: int = 90

    ollama_api_key: str = field(default="", repr=False)
    web_search_results: int = 5
    code_timeout: int = 10
    sandbox_image: str = "python:3.12-slim"
    mcp_timeout: int = 30
    files_read: list[Path] = field(default_factory=lambda: [Path(".")])
    files_write: list[Path] = field(default_factory=list)
    files_deny: list[str] = field(default_factory=list)
    max_tool_rounds: int = 5
    max_review_rounds: int = 2
    approval: str = "critical"
    log_path: Path = Path(".sensai/logs.db")
    log_content: bool = True


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


def _list_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    # Comma-separated, like "leak, notes": an empty value means none.
    return [item.strip() for item in raw.split(",") if item.strip()]


def _choice_env(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, got '{raw}'")
    return raw


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
        persona=os.getenv("SENSAI_PERSONA", Config.persona).strip(),
        personas_dir=Path(os.getenv("SENSAI_PERSONAS_DIR", "").strip() or Config.personas_dir),
        features=_features_env(),
        compact_threshold=_int_env("SENSAI_COMPACT_THRESHOLD", Config.compact_threshold, minimum=10, maximum=95),
        keep_recent_turns=_int_env("SENSAI_KEEP_RECENT_TURNS", Config.keep_recent_turns, minimum=1),
        sessions_dir=Path(os.getenv("SENSAI_SESSIONS_DIR", str(Config.sessions_dir))),
        profile_path=Path(os.getenv("SENSAI_PROFILE_PATH", str(Config.profile_path))),
        memory_path=Path(os.getenv("SENSAI_MEMORY_PATH", str(Config.memory_path))),
        guardrails_path=Path(os.getenv("SENSAI_GUARDRAILS_PATH", str(Config.guardrails_path))),
        mcp_path=Path(os.getenv("SENSAI_MCP_PATH", str(Config.mcp_path))),
        examples_path=Path(os.getenv("SENSAI_EXAMPLES_PATH", str(Config.examples_path))),
        examples_k=_int_env("SENSAI_EXAMPLES_K", Config.examples_k, minimum=0, maximum=5),
        prompts_dir=Path(os.getenv("SENSAI_PROMPTS_DIR", str(Config.prompts_dir))),
        prompt_cases_path=Path(os.getenv("SENSAI_PROMPT_CASES_PATH", str(Config.prompt_cases_path))),
        schedule_path=Path(os.getenv("SENSAI_SCHEDULE_PATH", str(Config.schedule_path))),
        reports_dir=Path(os.getenv("SENSAI_REPORTS_DIR", str(Config.reports_dir))),
        judge_model=os.getenv("SENSAI_JUDGE_MODEL", Config.judge_model).strip(),
        judge_max_tokens=_int_env("SENSAI_JUDGE_MAX_TOKENS", Config.judge_max_tokens, minimum=200, maximum=8000),
        embed_model=os.getenv("SENSAI_EMBED_MODEL", Config.embed_model),
        index_path=Path(os.getenv("SENSAI_INDEX_PATH", str(Config.index_path))),
        rag_top_k=_int_env("SENSAI_RAG_TOP_K", Config.rag_top_k, minimum=1, maximum=50),
        chunk_size=_int_env("SENSAI_RAG_CHUNK_SIZE", Config.chunk_size, minimum=100),
        chunk_overlap=_int_env("SENSAI_RAG_CHUNK_OVERLAP", Config.chunk_overlap, minimum=0),
        cache_path=Path(os.getenv("SENSAI_CACHE_PATH", str(Config.cache_path))),
        cache_threshold=_int_env("SENSAI_CACHE_THRESHOLD", Config.cache_threshold, minimum=50, maximum=100),
        ollama_api_key=os.getenv("OLLAMA_API_KEY", Config.ollama_api_key).strip(),
        web_search_results=_int_env("SENSAI_WEB_SEARCH_RESULTS", Config.web_search_results, minimum=1, maximum=10),
        code_timeout=_int_env("SENSAI_CODE_TIMEOUT", Config.code_timeout, minimum=1, maximum=60),
        sandbox_image=os.getenv("SENSAI_SANDBOX_IMAGE", Config.sandbox_image),
        mcp_timeout=_int_env("SENSAI_MCP_TIMEOUT", Config.mcp_timeout, minimum=1, maximum=300),
        files_read=[Path(path) for path in _list_env("SENSAI_FILES_READ", ["."])],
        files_write=[Path(path) for path in _list_env("SENSAI_FILES_WRITE", [])],
        files_deny=_list_env("SENSAI_FILES_DENY", []),
        max_tool_rounds=_int_env("SENSAI_MAX_TOOL_ROUNDS", Config.max_tool_rounds, minimum=1, maximum=20),
        max_review_rounds=_int_env("SENSAI_MAX_REVIEW_ROUNDS", Config.max_review_rounds, minimum=1, maximum=5),
        approval=_choice_env("SENSAI_APPROVAL", Config.approval, APPROVAL_MODES),
        log_path=Path(os.getenv("SENSAI_LOG_PATH", str(Config.log_path))),
        log_content=_bool_env("SENSAI_LOG_CONTENT") is not False,
    )
    if config.chunk_overlap >= config.chunk_size:
        raise ConfigError("SENSAI_RAG_CHUNK_OVERLAP must be smaller than SENSAI_RAG_CHUNK_SIZE")
    if config.features["tools"] and config.features["web_search"] and not config.ollama_api_key:
        raise ConfigError(f"SENSAI_FEATURE_WEB_SEARCH needs OLLAMA_API_KEY: {API_KEY_HELP}")
    return config
