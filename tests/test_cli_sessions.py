import pytest

from sensai.core.agent import Agent
from sensai.core.memory.conversation import Turn
from sensai.core.memory.profile import PROFILE_HEADER, Profile
from sensai.core.memory.sessions import SessionStore
from sensai.interfaces.cli.main import Session, chat_turn, handle_command, load_memory, parse_args


@pytest.fixture
def session(config, client):
    session = Session(config, Agent(config, client))
    load_memory(session, None)
    return session


def system_prompt(client) -> str:
    return client.chats[-1][0][0]["content"]


def test_every_turn_is_saved(session):
    chat_turn(session, "Who approved the transfer?")
    chat_turn(session, "Was there an invoice?")
    saved = SessionStore(session.config.sessions_dir).list()
    assert [(info.id, info.turns, info.title) for info in saved] == [
        (session.session_id, 2, "Who approved the transfer?")
    ]


def test_nothing_is_saved_before_the_first_message(session):
    chat_turn(session, "   ")
    assert session.session_id is None
    assert not session.config.sessions_dir.exists()


def test_nothing_is_saved_when_the_sessions_feature_is_off(session):
    session.config.features["sessions"] = False
    chat_turn(session, "Hi")
    assert not session.config.sessions_dir.exists()


def test_resume_restores_the_history_sent_to_the_model(config, client, capsys):
    first = Session(config, Agent(config, client))
    load_memory(first, None)
    chat_turn(first, "Who approved the transfer?")

    second = Session(config, Agent(config, client))
    load_memory(second, "last")
    chat_turn(second, "And the invoice?")

    assert f"Resumed conversation {first.session_id}: 1 turn" in capsys.readouterr().out
    assert [message["content"] for message in client.chats[-1][0]] == [
        "Who approved the transfer?", "Hello there ", "And the invoice?"
    ]
    assert second.session_id == first.session_id


def test_resume_by_id_from_the_chat(session, capsys):
    SessionStore(session.config.sessions_dir).save("alvor", session.agent.memory, "m")
    session.agent.memory.turns = [Turn("Who is Marc Vidal?", "The CFO.")]
    SessionStore(session.config.sessions_dir).save("vidal", session.agent.memory, "m")
    handle_command(session, "/new")
    handle_command(session, "/resume vidal")
    assert session.agent.memory.turns == [Turn("Who is Marc Vidal?", "The CFO.")]
    assert session.session_id == "vidal"


def test_resume_unknown_session_keeps_the_conversation(session, capsys):
    chat_turn(session, "Hi")
    turns = list(session.agent.memory.turns)
    handle_command(session, "/resume nope")
    assert session.agent.memory.turns == turns
    assert "No saved conversation 'nope'" in capsys.readouterr().err


def test_resume_without_any_saved_conversation(config, client, capsys):
    session = Session(config, Agent(config, client))
    load_memory(session, "last")
    assert "No saved conversation yet" in capsys.readouterr().out
    assert session.session_id is None


def test_new_starts_an_empty_conversation_and_keeps_the_old_one(session, capsys):
    chat_turn(session, "Hi")
    previous = session.session_id
    handle_command(session, "/new")
    assert session.agent.memory.turns == [] and session.session_id is None
    assert f"/resume {previous}" in capsys.readouterr().out
    chat_turn(session, "Another case")
    assert len(SessionStore(session.config.sessions_dir).list()) == 2


def test_sessions_lists_and_marks_the_current_one(session, capsys):
    chat_turn(session, "Who approved the transfer?")
    handle_command(session, "/sessions")
    line = capsys.readouterr().out.splitlines()[-2]
    assert line.startswith(f"* {session.session_id}") and line.endswith("Who approved the transfer?")


def test_sessions_commands_say_when_the_feature_is_off(session, capsys):
    session.config.features["sessions"] = False
    handle_command(session, "/sessions")
    assert "The sessions feature is off" in capsys.readouterr().out


def test_profile_is_sent_at_the_start_of_every_conversation(config, client):
    Profile(config.profile_path).add("Answer in French.")
    session = Session(config, Agent(config, client))
    load_memory(session, None)
    chat_turn(session, "Hi")
    assert system_prompt(client) == f"{PROFILE_HEADER}\n- Answer in French."


def test_profile_add_is_used_by_the_next_message(session, client, capsys):
    handle_command(session, "/profile add Never name the source.")
    chat_turn(session, "Hi")
    assert "Never name the source." in system_prompt(client)
    assert Profile(session.config.profile_path).entries == ["Never name the source."]
    assert "Added as instruction 1." in capsys.readouterr().out


def test_profile_remove(session, client, capsys):
    handle_command(session, "/profile add one")
    handle_command(session, "/profile add two")
    handle_command(session, "/profile remove 1")
    assert session.profile.entries == ["two"]
    chat_turn(session, "Hi")
    assert "one" not in system_prompt(client)


def test_profile_shows_the_numbered_instructions(session, capsys):
    handle_command(session, "/profile add Answer in French.")
    handle_command(session, "/profile")
    assert "  1. Answer in French." in capsys.readouterr().out


@pytest.mark.parametrize("argument", ["remove", "remove x", "remove 3", "add", "edit"])
def test_invalid_profile_command(session, capsys, argument):
    handle_command(session, f"/profile {argument}")
    assert capsys.readouterr().err
    assert session.profile.entries == []


def test_unreadable_profile_does_not_stop_the_chat(config, client, capsys):
    config.profile_path.write_bytes(b"\xff\xfe\xfa")
    session = Session(config, Agent(config, client))
    load_memory(session, None)
    chat_turn(session, "Hi")
    assert "The profile is not sent" in capsys.readouterr().err
    assert client.chats[-1][0][0]["role"] == "user"


@pytest.mark.parametrize(("argv", "expected"), [([], None), (["-r"], "last"), (["--resume", "alvor"], "alvor")])
def test_resume_option(argv, expected):
    assert parse_args(argv).resume == expected
