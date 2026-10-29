import os
from pathlib import Path

import pytest

from sensai.core.config import FEATURES, Config, ConfigError, load_config

VARIABLES = [
    "SENSAI_MODEL", "OLLAMA_HOST", "SENSAI_SYSTEM_PROMPT", "SENSAI_NUM_CTX", "SENSAI_THINK", "SENSAI_COMPACT_THRESHOLD",
    "SENSAI_KEEP_RECENT_TURNS", "SENSAI_SESSIONS_DIR", "SENSAI_PROFILE_PATH", "SENSAI_EMBED_MODEL",
    "SENSAI_PERSONA", "SENSAI_PERSONAS_DIR",
    "SENSAI_KEEP_RECENT_TURNS", "SENSAI_GUARDRAILS_PATH", "SENSAI_EMBED_MODEL", "SENSAI_PERSONA", "SENSAI_PERSONAS_DIR",
    "SENSAI_INDEX_PATH", "SENSAI_RAG_TOP_K", "SENSAI_RAG_CHUNK_SIZE", "SENSAI_RAG_CHUNK_OVERLAP",
    "SENSAI_TOOLS", "SENSAI_WEB_SEARCH", "OLLAMA_API_KEY", "SENSAI_WEB_SEARCH_RESULTS", "SENSAI_MAX_TOOL_ROUNDS",
    "SENSAI_APPROVAL",
    "SENSAI_CODE_TIMEOUT", "SENSAI_SANDBOX_IMAGE",
]


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in VARIABLES + [name for name in os.environ if name.startswith("SENSAI_FEATURE_")]:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    return path


def test_defaults(env_file):
    assert load_config(env_file) == Config()


def test_features_have_their_default(env_file):
    assert load_config(env_file).features == {
        "rag": True, "tools": True, "web_search": False, "code_exec": False,
        "sessions": True, "profile": True, "guardrails": True,
    }


def test_rag_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("SENSAI_INDEX_PATH", "/tmp/case.db")
    monkeypatch.setenv("SENSAI_RAG_TOP_K", "8")
    config = load_config(env_file)
    assert (config.embed_model, config.index_path, config.rag_top_k) == ("bge-m3", Path("/tmp/case.db"), 8)


def test_memory_files_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_SESSIONS_DIR", "/cases/alvor/sessions")
    monkeypatch.setenv("SENSAI_PROFILE_PATH", "/cases/profile.md")
    config = load_config(env_file)
    assert (config.sessions_dir, config.profile_path) == (Path("/cases/alvor/sessions"), Path("/cases/profile.md"))
def test_guardrails_path_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_GUARDRAILS_PATH", "/cases/alvor/guardrails.json")
    assert load_config(env_file).guardrails_path == Path("/cases/alvor/guardrails.json")


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


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("off", False), ("true", True), ("", None)])
@pytest.mark.parametrize("feature", list(FEATURES))
def test_feature_setting(env_file, monkeypatch, feature, value, expected):
    monkeypatch.setenv(f"SENSAI_FEATURE_{feature.upper()}", value)
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    default = FEATURES[feature].default
    assert load_config(env_file).features[feature] is (default if expected is None else expected)


def test_feature_from_env_file(env_file):
    env_file.write_text("SENSAI_FEATURE_RAG=false\n", encoding="utf-8")
    try:
        assert load_config(env_file).features["rag"] is False
    finally:
        os.environ.pop("SENSAI_FEATURE_RAG", None)


def test_unknown_feature_is_rejected(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_FEATURE_RAGG", "false")
    with pytest.raises(ConfigError, match="SENSAI_FEATURE_RAGG"):
        load_config(env_file)


def test_web_search_is_off_unless_enabled(env_file, monkeypatch):
    assert load_config(env_file).features["web_search"] is False
    monkeypatch.setenv("SENSAI_FEATURE_WEB_SEARCH", "true")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    config = load_config(env_file)
    assert config.features["web_search"] is True and config.ollama_api_key == "secret"
    assert "secret" not in repr(config)


def test_web_search_needs_an_api_key(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_FEATURE_WEB_SEARCH", "true")
    with pytest.raises(ConfigError, match="SENSAI_FEATURE_WEB_SEARCH needs OLLAMA_API_KEY"):
        load_config(env_file)


def test_disabled_tools_do_not_need_the_api_key(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_FEATURE_TOOLS", "false")
    monkeypatch.setenv("SENSAI_FEATURE_WEB_SEARCH", "true")
    assert load_config(env_file).features["tools"] is False


@pytest.mark.parametrize(
    ("old", "new"), [("SENSAI_TOOLS", "SENSAI_FEATURE_TOOLS"), ("SENSAI_WEB_SEARCH", "SENSAI_FEATURE_WEB_SEARCH")]
)
def test_renamed_settings_are_reported(env_file, monkeypatch, old, new):
    monkeypatch.setenv(old, "false")
    with pytest.raises(ConfigError, match=f"{old} is now {new}"):
        load_config(env_file)


def test_tool_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_MAX_TOOL_ROUNDS", "3")
    monkeypatch.setenv("SENSAI_WEB_SEARCH_RESULTS", "8")
    config = load_config(env_file)
    assert (config.max_tool_rounds, config.web_search_results) == (3, 8)


def test_approval_asks_for_critical_tools_by_default(env_file):
    assert load_config(env_file).approval == "critical"


@pytest.mark.parametrize(("value", "expected"), [("all", "all"), (" OFF ", "off"), ("Critical", "critical")])
def test_approval_setting(env_file, monkeypatch, value, expected):
    monkeypatch.setenv("SENSAI_APPROVAL", value)
    assert load_config(env_file).approval == expected


def test_unknown_approval_lists_the_valid_values(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_APPROVAL", "sometimes")
    with pytest.raises(ConfigError, match="critical, all, off"):
        load_config(env_file)
def test_code_exec_settings_from_environment(env_file, monkeypatch):
    assert load_config(env_file).features["code_exec"] is False
    monkeypatch.setenv("SENSAI_FEATURE_CODE_EXEC", "true")
    monkeypatch.setenv("SENSAI_CODE_TIMEOUT", "30")
    monkeypatch.setenv("SENSAI_SANDBOX_IMAGE", "python:3.13-slim")
    config = load_config(env_file)
    assert (config.features["code_exec"], config.code_timeout, config.sandbox_image) == (True, 30, "python:3.13-slim")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SENSAI_THINK", "maybe"),
        ("SENSAI_FEATURE_RAG", "maybe"),
        ("SENSAI_FEATURE_TOOLS", "maybe"),
        ("SENSAI_RAG_TOP_K", "five"),
        ("SENSAI_RAG_TOP_K", "0"),
        ("SENSAI_RAG_TOP_K", "5000"),
        ("SENSAI_RAG_CHUNK_OVERLAP", "5000"),
        ("SENSAI_COMPACT_THRESHOLD", "5"),
        ("SENSAI_COMPACT_THRESHOLD", "100"),
        ("SENSAI_KEEP_RECENT_TURNS", "0"),
        ("SENSAI_FEATURE_WEB_SEARCH", "maybe"),
        ("SENSAI_MAX_TOOL_ROUNDS", "0"),
        ("SENSAI_MAX_TOOL_ROUNDS", "50"),
        ("SENSAI_WEB_SEARCH_RESULTS", "0"),
        ("SENSAI_WEB_SEARCH_RESULTS", "11"),
        ("SENSAI_FEATURE_CODE_EXEC", "maybe"),
        ("SENSAI_CODE_TIMEOUT", "0"),
        ("SENSAI_CODE_TIMEOUT", "61"),
    ],
)
def test_invalid_values_are_rejected(env_file, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        load_config(env_file)
