from pathlib import Path

import pytest

from sensai.core.config import Config, ConfigError, load_config

VARIABLES = [
    "SENSAI_MODEL", "OLLAMA_HOST", "SENSAI_SYSTEM_PROMPT", "SENSAI_NUM_CTX", "SENSAI_THINK", "SENSAI_COMPACT_THRESHOLD",
    "SENSAI_KEEP_RECENT_TURNS", "SENSAI_EMBED_MODEL",
    "SENSAI_INDEX_PATH", "SENSAI_RAG_TOP_K", "SENSAI_RAG_CHUNK_SIZE", "SENSAI_RAG_CHUNK_OVERLAP",
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


def test_budget_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_COMPACT_THRESHOLD", "60")
    monkeypatch.setenv("SENSAI_KEEP_RECENT_TURNS", "2")
    config = load_config(env_file)
    assert (config.compact_threshold, config.keep_recent_turns) == (60, 2)


@pytest.mark.parametrize(("value", "expected"), [("", None), ("false", False), ("0", False), ("True", True)])
def test_think_setting(env_file, monkeypatch, value, expected):
    monkeypatch.setenv("SENSAI_THINK", value)
    assert load_config(env_file).think is expected


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
    ],
)
def test_invalid_values_are_rejected(env_file, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        load_config(env_file)
