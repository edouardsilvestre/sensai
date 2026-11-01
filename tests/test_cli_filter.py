import pytest

from sensai.core.agent import Agent
from sensai.core.memory.sessions import SessionStore
from sensai.core.rag.models import SearchFilter
from sensai.core.rag.store import IndexStats
from sensai.interfaces.cli.main import Session, handle_command, load_memory


class FakeIndex:
    def __init__(self):
        self.counted: list[list[SearchFilter]] = []

    def count_documents(self, filters) -> int:
        self.counted.append(list(filters))
        return 12

    def stats(self) -> IndexStats:
        return IndexStats(340, 5000, "m", 256)


@pytest.fixture
def session(config, client):
    session = Session(config, Agent(config, client))
    session.index = FakeIndex()
    return session


def test_a_filter_limits_the_searches(session, capsys):
    handle_command(session, '/filter after:2023-03 from:"Jean Dupont" kind:email')
    assert session.agent.search_filter == SearchFilter(after="2023-03", author="Jean Dupont", kind="email")
    assert session.index.counted == [[session.agent.search_filter]]
    out = capsys.readouterr().out
    assert 'Filter: after:2023-03 from:"Jean Dupont" kind:email' in out
    assert "12 of the 340 indexed documents" in out


def test_the_filter_is_shown_and_cleared(session, capsys):
    handle_command(session, "/filter kind:pdf")
    capsys.readouterr()
    handle_command(session, "/filter")
    assert "Filter: kind:pdf" in capsys.readouterr().out
    handle_command(session, "/filter clear")
    assert session.agent.search_filter.is_empty()
    assert "No filter: the searches look at every document." in capsys.readouterr().out


def test_a_wrong_filter_keeps_the_current_one(session, capsys):
    handle_command(session, "/filter kind:pdf")
    handle_command(session, "/filter after:yesterday")
    assert session.agent.search_filter == SearchFilter(kind="pdf")
    err = capsys.readouterr().err
    assert "after must be a year, a month or a day" in err
    assert "Usage: /filter [clear | after:2023-03 before:2023-07 from:NAME to:NAME kind:KIND source:FOLDER/]" in err


def test_a_filter_without_index(config, client, capsys):
    session = Session(config, Agent(config, client))
    handle_command(session, "/filter kind:pdf")
    assert session.agent.search_filter == SearchFilter(kind="pdf")
    assert "Filter: kind:pdf." in capsys.readouterr().out


def test_a_new_conversation_clears_the_filter(session, capsys):
    handle_command(session, "/filter from:Marc after:2023")
    capsys.readouterr()
    handle_command(session, "/new")
    assert session.agent.search_filter.is_empty()
    assert "Filter after:2023 from:Marc cleared: /filter to set it again." in capsys.readouterr().out


def test_a_resumed_conversation_clears_the_filter(session, capsys):
    load_memory(session, None)
    SessionStore(session.config.sessions_dir).save("vidal", session.agent.memory, "m")
    handle_command(session, "/filter kind:pdf")
    capsys.readouterr()
    handle_command(session, "/resume vidal")
    assert session.agent.search_filter.is_empty()
    assert "Filter kind:pdf cleared" in capsys.readouterr().out


def test_no_filter_cleared_without_a_filter(session, capsys):
    handle_command(session, "/new")
    assert "Filter" not in capsys.readouterr().out
