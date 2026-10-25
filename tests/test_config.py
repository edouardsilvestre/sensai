import os
from pathlib import Path

import pytest

from sensai.core.config import FEATURES, Config, ConfigError, load_config

VARIABLES = [
    "SENSAI_MODEL", "OLLAMA_HOST", "SENSAI_SYSTEM_PROMPT", "SENSAI_NUM_CTX", "SENSAI_THINK", "SENSAI_EMBED_MODEL",
    "SENSAI_INDEX_PATH", "SENSAI_RAG_TOP_K", "SENSAI_RAG_CHUNK_SIZE", "SENSAI_RAG_CHUNK_OVERLAP",
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


def test_every_feature_is_on_by_default(env_file):
    assert load_config(env_file).features == {name: True for name in FEATURES}


def test_rag_settings_from_environment(env_file, monkeypatch):
    monkeypatch.setenv("SENSAI_EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("SENSAI_INDEX_PATH", "/tmp/case.db")
    monkeypatch.setenv("SENSAI_RAG_TOP_K", "8")
    config = load_config(env_file)
    assert (config.embed_model, config.index_path, config.rag_top_k) == ("bge-m3", Path("/tmp/case.db"), 8)


@pytest.mark.parametrize(("value", "expected"), [("", None), ("false", False), ("0", False), ("True", True)])
def test_think_setting(env_file, monkeypatch, value, expected):
    monkeypatch.setenv("SENSAI_THINK", value)
    assert load_config(env_file).think is expected


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("off", False), ("true", True), ("", True)])
def test_feature_setting(env_file, monkeypatch, value, expected):
    monkeypatch.setenv("SENSAI_FEATURE_RAG", value)
    assert load_config(env_file).features["rag"] is expected


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SENSAI_THINK", "maybe"),
        ("SENSAI_FEATURE_RAG", "maybe"),
        ("SENSAI_RAG_TOP_K", "five"),
        ("SENSAI_RAG_TOP_K", "0"),
        ("SENSAI_RAG_TOP_K", "5000"),
        ("SENSAI_RAG_CHUNK_OVERLAP", "5000"),
    ],
)
def test_invalid_values_are_rejected(env_file, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        load_config(env_file)
