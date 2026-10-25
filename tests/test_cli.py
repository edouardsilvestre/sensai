import pytest

from sensai.core.agent import Agent
from sensai.interfaces.cli.main import Session, handle_command, run_status


@pytest.fixture
def session(config, client):
    return Session(config, Agent(config, client))


def test_features_lists_every_feature(session, capsys):
    handle_command(session, "/features")
    assert capsys.readouterr().out.split() == ["rag", "on", "answer", "from", "the", "indexed", "documents"]


def test_feature_can_be_turned_off_and_on(session, capsys):
    handle_command(session, "/feature rag off")
    assert session.config.features["rag"] is False
    handle_command(session, "/feature RAG on")
    assert session.config.features["rag"] is True
    assert "No documents indexed yet" in capsys.readouterr().out


@pytest.mark.parametrize("argument", ["", "rag", "rag maybe", "unknown off"])
def test_invalid_feature_command(session, capsys, argument):
    handle_command(session, f"/feature {argument}")
    assert session.config.features["rag"] is True
    assert "Usage: /feature NAME on|off" in capsys.readouterr().err


def test_status_shows_the_features(config, capsys):
    config.features["rag"] = False
    run_status(config)
    assert "Features: rag off" in capsys.readouterr().out
