from pathlib import Path

import pytest

from sensai.core.config import Config, ConfigError, load_config

VARIABLES = [
    "SENSAI_MODEL", "OLLAMA_HOST", "SENSAI_SYSTEM_PROMPT", "SENSAI_NUM_CTX", "SENSAI_THINK", "SENSAI_COMPACT_THRESHOLD",
    "SENSAI_KEEP_RECENT_TURNS", "SENSAI_EMBED_MODEL", "SENSAI_PERSONA", "SENSAI_PERSONAS_DIR",
    "SENSAI_INDEX_PATH", "SENSAI_RAG_TOP_K", "SENSAI_RAG_CHUNK_SIZE", "SENSAI_RAG_CHUNK_OVERLAP",
    "SENSAI_TOOLS", "SENSAI_WEB_SEARCH", "OLLAMA_API_KEY", "SENSAI_WEB_SEARCH_RESULTS", "SENSAI_MAX_TOOL_ROUNDS",
]


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    return path


def test_defaults(env_file):
    assert load_config(env_file) == Config()


def test_rag_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("SENSAI_INDEX_PATH", "/tmp/case.db")
    monkeypatch.setenv("SENSAI_RAG_TOP_K", "8")
    config = load_config(env_file)
    assert (config.embed_model, config.index_path, config.rag_top_k) == ("bge-m3", Path("/tmp/case.db"), 8)


def test_persona_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_PERSONA", " editor ")
    monkeypatch.setenv("SENSAI_PERSONAS_DIR", "/tmp/newsroom")
    config = load_config(env_file)
    assert (config.persona, config.personas_dir) == ("editor", Path("/tmp/newsroom"))


def test_empty_persona_settings(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_PERSONA", "")
    monkeypatch.setenv("SENSAI_PERSONAS_DIR", "")
    config = load_config(env_file)
    assert (config.persona, config.personas_dir) == ("", Config.personas_dir)


def test_budget_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_COMPACT_THRESHOLD", "60")
    monkeypatch.setenv("SENSAI_KEEP_RECENT_TURNS", "2")
    config = load_config(env_file)
    assert (config.compact_threshold, config.keep_recent_turns) == (60, 2)


@pytest.mark.parametrize(("value", "expected"), [("", None), ("false", False), ("0", False), ("True", True)])
def test_think_setting(env_file, monkeypatch, value, expected):
    monkeypatch.setenv("SENSAI_THINK", value)
    assert load_config(env_file).think is expected


def test_web_search_is_off_unless_enabled(env_file, monkeypatch):
    assert load_config(env_file).web_search is False
    monkeypatch.setenv("SENSAI_WEB_SEARCH", "true")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    config = load_config(env_file)
    assert config.web_search is True and config.ollama_api_key == "secret"
    assert "secret" not in repr(config)


def test_web_search_needs_an_api_key(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_WEB_SEARCH", "true")
    with pytest.raises(ConfigError, match="OLLAMA_API_KEY"):
        load_config(env_file)


@pytest.mark.parametrize(("value", "expected"), [("", True), ("true", True), ("false", False)])
def test_tools_setting(env_file, monkeypatch, value, expected):
    monkeypatch.setenv("SENSAI_TOOLS", value)
    assert load_config(env_file).tools is expected


def test_disabled_tools_do_not_need_the_api_key(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_TOOLS", "false")
    monkeypatch.setenv("SENSAI_WEB_SEARCH", "true")
    assert load_config(env_file).tools is False


def test_tool_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_MAX_TOOL_ROUNDS", "3")
    monkeypatch.setenv("SENSAI_WEB_SEARCH_RESULTS", "8")
    config = load_config(env_file)
    assert (config.max_tool_rounds, config.web_search_results) == (3, 8)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SENSAI_THINK", "maybe"),
        ("SENSAI_RAG_TOP_K", "five"),
        ("SENSAI_RAG_TOP_K", "0"),
        ("SENSAI_RAG_TOP_K", "5000"),
        ("SENSAI_RAG_CHUNK_OVERLAP", "5000"),
        ("SENSAI_COMPACT_THRESHOLD", "5"),
        ("SENSAI_COMPACT_THRESHOLD", "100"),
        ("SENSAI_KEEP_RECENT_TURNS", "0"),
        ("SENSAI_WEB_SEARCH", "maybe"),
        ("SENSAI_TOOLS", "maybe"),
        ("SENSAI_MAX_TOOL_ROUNDS", "0"),
        ("SENSAI_MAX_TOOL_ROUNDS", "50"),
        ("SENSAI_WEB_SEARCH_RESULTS", "0"),
        ("SENSAI_WEB_SEARCH_RESULTS", "11"),
    ],
)
def test_invalid_values_are_rejected(env_file, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        load_config(env_file)
