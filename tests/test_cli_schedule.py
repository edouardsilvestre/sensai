from datetime import datetime, timedelta

import pytest

from conftest import FakeClient
from sensai.core.agent import Agent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.memory.conversation import Turn
from sensai.core.memory.sessions import SessionStore
from sensai.core.rag.models import SearchResult
from sensai.core.schedule import Schedule
from sensai.core.tools.registry import ToolRegistry, requires_approval
from sensai.interfaces.cli.main import (
    HELP,
    Session,
    feature_command,
    handle_command,
    load_memory,
    load_schedule,
    main,
    parse_args,
    print_schedule_news,
    run_due_tasks,
    watch_schedule,
)

PAST = datetime(2026, 1, 1, 8, 0)


@requires_approval("the query leaves the machine")
def send(query: str) -> str:
    """
    Send a query.

    Args:
      query: The query

    Returns:
      The result
    """
    return f"sent {query}"


def excerpt(chunk_id: int, text: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, location="p. 1", source="leak.pdf", score=0.9, description="")


class Excerpts:
    def __init__(self):
        self.found = [excerpt(1, "Orion statutes")]

    def search(self, query: str, k: int) -> list[SearchResult]:
        return self.found


class DownClient(FakeClient):
    def chat_stream(self, *args, **kwargs):
        raise LLMError("Ollama is not running")
        yield


@pytest.fixture
def session(config, client):
    session = Session(config, Agent(config, client, tools=ToolRegistry([send])))
    load_memory(session, None)
    session.schedule = load_schedule(config)
    return session


def source_rule(action: str) -> Guardrails:
    return Guardrails(parse_rules([{"name": "source", "terms": ["Claire Benoit"], "on": ["input"], "action": action}]))


def due_task(session, question="Any news?", allow=()):
    """A task added long ago, so that it is due now."""
    return session.schedule.add("every 1h", question, list(allow), PAST)


def test_add_a_task(session, capsys):
    handle_command(session, "/schedule add daily 08:00 Any news about Orion?")
    task = session.schedule.tasks[0]
    assert (task.when, task.question, task.allow) == ("daily 08:00", "Any news about Orion?", [])
    out = capsys.readouterr().out
    assert "Task 1 added, first run on " in out and "python -m sensai schedule" in out


def test_add_a_task_allowed_to_use_a_tool(session):
    handle_command(session, "/schedule add weekly MON 08:00 --allow send Any news?")
    task = session.schedule.tasks[0]
    assert (task.when, task.allow, task.question) == ("weekly mon 08:00", ["send"], "Any news?")


@pytest.mark.parametrize("argument", [
    "add", "add daily 08:00", "add Any news?", "add every 1h --allow send", "add every 1h --allow",
    "add sometimes Any news?", "remove", "remove x", "remove ²", "pause 1",
])
def test_invalid_schedule_command(session, capsys, argument):
    handle_command(session, f"/schedule {argument}")
    assert capsys.readouterr().err
    assert session.schedule.tasks == []


def test_invalid_time_says_what_is_accepted(session, capsys):
    handle_command(session, "/schedule add daily 25:00 Any news?")
    assert "every 30m, every 6h" in capsys.readouterr().err
    assert session.schedule.tasks == []


def test_only_the_tools_enabled_can_be_allowed(session, capsys):
    handle_command(session, "/schedule add every 1h --allow send,web_search Any news?")
    assert "Cannot allow web_search" in capsys.readouterr().err
    assert session.schedule.tasks == []


def test_a_task_goes_through_the_guardrails_when_it_is_added(session, capsys):
    session.agent.guardrails = source_rule("redact")
    handle_command(session, "/schedule add every 1h Where is Claire Benoit?")
    assert "Claire Benoit" not in session.config.schedule_path.read_text(encoding="utf-8")
    assert "[guardrail] source" in capsys.readouterr().out


def test_a_blocked_task_is_not_added(session, capsys):
    session.agent.guardrails = source_rule("block")
    handle_command(session, "/schedule add every 1h Where is Claire Benoit?")
    assert not session.config.schedule_path.exists()
    assert "blocked by the guardrail source" in capsys.readouterr().err


def test_schedule_lists_the_tasks(session, capsys):
    handle_command(session, "/schedule add daily 08:00 --allow send Any news?")
    capsys.readouterr()
    handle_command(session, "/schedule")
    out = capsys.readouterr().out
    assert "  1. Any news?" in out
    assert "daily 08:00, next run " in out and "not run yet, allowed: send" in out and "(late)" not in out


def test_schedule_shows_the_late_tasks(session, capsys):
    due_task(session)
    handle_command(session, "/schedule")
    out = capsys.readouterr().out
    assert "(late)" in out and "Late tasks run as soon as python -m sensai schedule is started." in out


def test_schedule_when_empty(session, capsys):
    handle_command(session, "/schedule")
    assert "No scheduled task." in capsys.readouterr().out


def test_remove_a_task(session, capsys):
    due_task(session)
    handle_command(session, "/schedule remove 1")
    assert "Removed: Any news?" in capsys.readouterr().out
    assert session.schedule.tasks == []


def test_remove_a_task_that_does_not_exist(session, capsys):
    handle_command(session, "/schedule remove 2")
    assert "No task 2" in capsys.readouterr().err


def test_the_commands_say_when_the_feature_is_off(session, capsys):
    session.config.features["schedule"] = False
    handle_command(session, "/schedule")
    handle_command(session, "/reports")
    assert capsys.readouterr().out.count("The schedule feature is off") == 2


def test_turning_the_feature_on_opens_the_schedule(config, client, capsys):
    config.features["schedule"] = False
    session = Session(config, Agent(config, client))
    feature_command(session, "schedule on")
    assert session.schedule is not None


def test_an_invalid_file_is_reported_and_kept(config, client, capsys):
    config.schedule_path.write_text("not json", encoding="utf-8")
    session = Session(config, Agent(config, client))
    session.schedule = load_schedule(config)
    assert "Schedule disabled" in capsys.readouterr().err
    handle_command(session, "/schedule add every 1h Any news?")
    assert "could not be read" in capsys.readouterr().out
    assert config.schedule_path.read_text(encoding="utf-8") == "not json"


def test_a_due_task_writes_a_report(session, capsys):
    task = due_task(session)
    assert run_due_tasks(session)
    assert "New report: " in capsys.readouterr().out
    assert session.schedule.tasks[0].status == "report written"
    assert session.schedule.reports[0].question == "Any news?"
    assert (session.config.reports_dir / session.schedule.reports[0].file).exists()
    assert datetime.fromisoformat(task.next_run) < datetime.fromisoformat(session.schedule.tasks[0].next_run)


def test_a_task_runs_once_per_slot(session, client):
    due_task(session)
    run_due_tasks(session)
    run_due_tasks(session)
    assert len(client.chats) == 1


def test_a_task_not_due_does_not_run(session, client):
    session.schedule.add("every 1h", "Any news?", [], datetime.now())
    run_due_tasks(session)
    assert client.chats == []


def run_again(session):
    """Makes every task due again and runs them."""
    schedule = session.schedule
    for task in schedule.tasks:
        task.next_run = PAST.isoformat()
    schedule._save()
    return run_due_tasks(session)


def test_a_run_that_finds_the_same_writes_no_report(session, capsys):
    session.agent.retriever = Excerpts()
    due_task(session)
    run_due_tasks(session)
    run_again(session)
    assert "Nothing new." in capsys.readouterr().out
    assert len(session.schedule.reports) == 1
    assert session.schedule.tasks[0].status == "nothing new"


def test_a_run_that_finds_something_else_writes_a_report(session, capsys):
    session.agent.retriever = excerpts = Excerpts()
    due_task(session)
    run_due_tasks(session)
    run_again(session)
    excerpts.found = excerpts.found + [excerpt(2, "Vidal resigned")]
    run_again(session)
    assert len(session.schedule.reports) == 2
    assert session.schedule.tasks[0].status == "report written"
    run_again(session)
    assert len(session.schedule.reports) == 2


def test_a_failed_run_keeps_the_evidence_of_the_last_one(session, client, monkeypatch):
    session.agent.retriever = Excerpts()
    task = due_task(session)
    run_due_tasks(session)
    evidence = session.schedule.tasks[0].evidence
    assert evidence
    monkeypatch.setattr(client, "chat_stream", DownClient().chat_stream)
    run_again(session)
    assert session.schedule.tasks[0].evidence == evidence and session.schedule.tasks[0].id == task.id


def test_a_refused_tool_gives_a_report_even_with_the_same_evidence(session, client, capsys):
    session.agent.retriever = Excerpts()
    due_task(session)
    run_due_tasks(session)
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    run_again(session)
    assert "A tool could not run, send: this scheduled task was not allowed" in capsys.readouterr().err
    assert len(session.schedule.reports) == 2
    report = session.config.reports_dir / session.schedule.reports[1].file
    assert "## Tools that could not run" in report.read_text(encoding="utf-8")


def test_a_failed_run_is_reported_and_the_next_task_still_runs(config, capsys):
    session = Session(config, Agent(config, DownClient(), tools=ToolRegistry([send])))
    load_memory(session, None)
    session.schedule = load_schedule(config)
    due_task(session, "First")
    due_task(session, "Second")
    assert not run_due_tasks(session)
    assert capsys.readouterr().err.count("Ollama is not running") == 2
    assert [task.failed for task in session.schedule.tasks] == [True, True]
    assert session.schedule.tasks[0].status == "failed: Ollama is not running"
    assert datetime.fromisoformat(session.schedule.tasks[0].next_run) > datetime.now()


def test_a_run_after_a_failure_clears_it(session):
    task = due_task(session)
    session.schedule.finish(task, "failed: Ollama is not running", failed=True)
    run_due_tasks(session)
    assert session.schedule.tasks[0].failed is False


def test_ctrl_c_during_a_run_is_recorded_and_stops_the_scheduler(session, client, capsys, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
        yield

    monkeypatch.setattr(client, "chat_stream", interrupted)
    due_task(session)
    assert watch_schedule(session, once=False) == 0
    assert session.schedule.tasks[0].status == "failed: interrupted"
    assert "[interrupted]" in capsys.readouterr().out


def test_the_scheduler_waits_and_sees_the_tasks_added_meanwhile(session, client, monkeypatch):
    polls = []

    def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 1:
            Schedule(session.config.schedule_path, session.config.reports_dir).add("every 1h", "Any news?", [], PAST)
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr("sensai.interfaces.cli.main.time.sleep", sleep)
    assert watch_schedule(session, once=False) == 0
    assert len(client.chats) == 1


def test_once_runs_what_is_due_and_exits(session, client, monkeypatch):
    monkeypatch.setattr("sensai.interfaces.cli.main.time.sleep", lambda seconds: pytest.fail("it should exit"))
    due_task(session)
    assert watch_schedule(session, once=True) == 0
    assert len(client.chats) == 1


def test_once_says_when_a_task_failed(config, capsys):
    session = Session(config, Agent(config, DownClient()))
    load_memory(session, None)
    session.schedule = load_schedule(config)
    due_task(session)
    assert watch_schedule(session, once=True) == 1


def test_a_file_broken_while_running_does_not_stop_the_scheduler(session, capsys, monkeypatch):
    def sleep(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("sensai.interfaces.cli.main.time.sleep", sleep)
    session.config.schedule_path.write_text("not json", encoding="utf-8")
    assert watch_schedule(session, once=False) == 0
    assert "Invalid schedule file" in capsys.readouterr().err


def test_reports_lists_the_unread_ones_first(session, capsys):
    due_task(session)
    run_due_tasks(session)
    capsys.readouterr()
    handle_command(session, "/reports")
    assert "  * 1. " in capsys.readouterr().out
    handle_command(session, "/reports 1")
    assert capsys.readouterr().out.startswith("# Any news?")
    handle_command(session, "/reports")
    assert "    1. " in capsys.readouterr().out


def test_reports_when_empty(session, capsys):
    handle_command(session, "/reports")
    assert "No report yet" in capsys.readouterr().out


@pytest.mark.parametrize("argument", ["x", "2", "²"])
def test_invalid_report_number(session, capsys, argument):
    handle_command(session, f"/reports {argument}")
    assert capsys.readouterr().err


def test_the_chat_tells_about_new_reports_and_failures(session, capsys):
    task = due_task(session)
    run_due_tasks(session)
    session.schedule.finish(task, "failed: Ollama is not running", failed=True)
    capsys.readouterr()
    print_schedule_news(session)
    captured = capsys.readouterr()
    assert "[schedule] 1 new report. /reports to read them." in captured.out
    assert "1 scheduled task failed" in captured.err


def test_the_chat_says_nothing_when_nothing_happened(session, capsys):
    print_schedule_news(session)
    assert capsys.readouterr() == ("", "")


def test_a_task_conversation_can_be_resumed_but_is_never_the_last_one(session, capsys):
    task = due_task(session)
    run_due_tasks(session)
    store = SessionStore(session.config.sessions_dir)
    assert store.latest() is None and store.list()[0].id == task.id
    handle_command(session, f"/resume {task.id}")
    assert session.agent.memory.turns[0].user == "Any news?"


def test_the_last_conversation_is_found_behind_a_task(session):
    store = SessionStore(session.config.sessions_dir)
    session.agent.memory.turns = [Turn("Mine", "Answer")]
    store.save("20261102-080000", session.agent.memory, "m")
    task = due_task(session)
    session.agent.memory.turns = []
    run_due_tasks(session)
    assert store.list()[0].id == task.id and store.latest() == "20261102-080000"


def test_the_scheduler_refuses_to_start_when_the_feature_is_off(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text("SENSAI_FEATURE_SCHEDULE=false\n", encoding="utf-8")
    monkeypatch.delenv("SENSAI_FEATURE_SCHEDULE", raising=False)
    try:
        assert main(["-e", str(env), "schedule"]) == 1
    finally:
        monkeypatch.delenv("SENSAI_FEATURE_SCHEDULE", raising=False)
    assert "The schedule feature is off" in capsys.readouterr().err


@pytest.mark.parametrize(("argv", "once"), [(["schedule"], False), (["schedule", "--once"], True)])
def test_schedule_arguments(argv, once):
    args = parse_args(argv)
    assert args.command == "schedule" and args.once is once


def test_help_lists_the_schedule_commands():
    assert "/schedule add WHEN TASK" in HELP and "/reports [N]" in HELP


def test_a_future_task_is_not_late(session, capsys):
    session.schedule.add("every 1h", "Any news?", [], datetime.now() + timedelta(hours=1))
    handle_command(session, "/schedule")
    assert "(late)" not in capsys.readouterr().out
