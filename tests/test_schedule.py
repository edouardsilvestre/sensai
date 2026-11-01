import threading
from datetime import datetime

import pytest

from conftest import FakeClient
from sensai.core.agent import Agent
from sensai.core.document.document import Document
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import ToolCall
from sensai.core.memory.conversation import Turn
from sensai.core.memory.sessions import SessionStore
from sensai.core.persona import load_personas
from sensai.core.rag.models import SearchResult
from sensai.core.schedule import Run, Schedule, ScheduleError, next_run, run_task
from sensai.core.tools.registry import ToolError, ToolRegistry, requires_approval

# A Monday.
MONDAY = datetime(2026, 11, 2, 9, 30)
LATER = datetime(2026, 11, 2, 18, 0)


def excerpt(chunk_id: int, text: str, source: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, location="p. 1", source=source, score=0.9, description="")


class Excerpts:
    EXCERPT = excerpt(1, "Orion statutes", "statutes.pdf")
    OTHER = excerpt(2, "Vidal resigned", "minutes.pdf")

    def __init__(self):
        self.found = [self.EXCERPT]

    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
        self.query = query
        return self.found


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


@pytest.fixture
def schedule(config):
    return Schedule(config.schedule_path, config.reports_dir)


@pytest.fixture
def sessions(config):
    return SessionStore(config.sessions_dir)


@pytest.mark.parametrize(("when", "expected"), [
    ("every 30m", datetime(2026, 11, 2, 10, 0)),
    ("every 6h", datetime(2026, 11, 2, 15, 30)),
    ("EVERY 2d", datetime(2026, 11, 4, 9, 30)),
    ("daily 10:00", datetime(2026, 11, 2, 10, 0)),
    ("daily 08:00", datetime(2026, 11, 3, 8, 0)),
    ("daily 09:30", datetime(2026, 11, 3, 9, 30)),
    ("weekly wed 08:00", datetime(2026, 11, 4, 8, 0)),
    ("weekly mon 10:00", datetime(2026, 11, 2, 10, 0)),
    ("weekly mon 08:00", datetime(2026, 11, 9, 8, 0)),
    ("weekly sun 23:59", datetime(2026, 11, 8, 23, 59)),
])
def test_next_run(when, expected):
    assert next_run(when, MONDAY) == expected


@pytest.mark.parametrize("when", [
    "", "every", "every 0m", "every 5", "every 5s", "every -1h", "every ²h", "every 99999999999999d", "daily",
    "daily 25:00", "daily 08:61", "weekly 08:00", "weekly lun 08:00", "weekly mon", "monthly 08:00", "every 1h daily",
])
def test_invalid_schedule(when):
    with pytest.raises(ScheduleError, match="Invalid schedule"):
        next_run(when, MONDAY)


def test_tasks_are_kept_in_the_file(config, schedule):
    task = schedule.add("Daily  08:00", "Any news?", ["web_search"], MONDAY)
    assert (task.when, task.next_run, task.id) == ("daily 08:00", "2026-11-03T08:00:00", "task-20261102-093000")
    assert Schedule(config.schedule_path, config.reports_dir).tasks == [task]


def test_two_tasks_added_at_the_same_time_have_their_own_id(schedule):
    assert schedule.add("every 1h", "one", [], MONDAY).id != schedule.add("every 1h", "two", [], MONDAY).id


def test_a_task_added_elsewhere_is_not_lost(config, schedule):
    Schedule(config.schedule_path, config.reports_dir).add("every 1h", "from the chat", [], MONDAY)
    schedule.add("every 1h", "from here", [], MONDAY)
    assert [task.question for task in schedule.tasks] == ["from the chat", "from here"]


def test_remove(schedule):
    schedule.add("every 1h", "one", [], MONDAY)
    schedule.add("every 1h", "two", [], MONDAY)
    assert schedule.remove(1).question == "one"
    assert [task.question for task in schedule.tasks] == ["two"]
    with pytest.raises(ScheduleError, match="No task 3"):
        schedule.remove(3)


@pytest.mark.parametrize("content", [
    "not json", "[]", '{"tasks": []}', '{"tasks": [{"id": "x"}], "reports": []}',
    '{"tasks": [{"id": "x", "when": "sometimes", "question": "q", "next_run": "2026-11-02T08:00:00"}], "reports": []}',
    '{"tasks": [{"id": "x", "when": "every 1h", "question": "q", "next_run": "soon"}], "reports": []}',
    '{"tasks": [{"id": "x", "when": "every 1h", "question": " ", "next_run": "2026-11-02T08:00:00"}], "reports": []}',
    '{"tasks": [{"id": "x", "when": "every 1h", "question": "q", "next_run": 5}], "reports": []}',
])
def test_invalid_file_is_refused_and_not_overwritten(config, content):
    config.schedule_path.write_text(content, encoding="utf-8")
    with pytest.raises(ScheduleError, match="Invalid schedule file"):
        Schedule(config.schedule_path, config.reports_dir)
    assert config.schedule_path.read_text(encoding="utf-8") == content


def test_due_only_lists_the_tasks_whose_time_has_come(schedule):
    schedule.add("every 1h", "soon", [], MONDAY)
    schedule.add("daily 08:00", "tomorrow", [], MONDAY)
    assert [task.question for task in schedule.due(LATER)] == ["soon"]
    assert schedule.due(MONDAY) == []


def test_a_task_missed_many_times_runs_once(schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    assert schedule.claim(task, LATER)
    assert schedule.due(LATER) == []
    assert schedule.tasks[0].next_run == "2026-11-02T19:00:00"
    assert schedule.tasks[0].last_run == "2026-11-02T18:00:00"


def in_parallel(work, count: int) -> None:
    """Runs `work(number)` in `count` threads, each failure failing the test."""
    failures = []

    def run(number: int) -> None:
        try:
            work(number)
        except Exception as e:
            failures.append(e)

    threads = [threading.Thread(target=run, args=(number,)) for number in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_no_task_is_lost_when_several_write_at_once(config, schedule):
    def add(number: int) -> None:
        other = Schedule(config.schedule_path, config.reports_dir)
        for task in range(10):
            other.add("every 1h", f"{number}-{task}", [], MONDAY)

    in_parallel(add, 6)
    schedule.reload()
    assert len({task.question for task in schedule.tasks}) == 60
    assert len({task.id for task in schedule.tasks}) == 60


def test_reading_during_a_write_does_not_stop_the_write(config, schedule):
    def work(number: int) -> None:
        other = Schedule(config.schedule_path, config.reports_dir)
        for task in range(20):
            other.add("every 1h", "Any news?", [], MONDAY) if number == 0 else other.reload()

    in_parallel(work, 4)
    schedule.reload()
    assert len(schedule.tasks) == 20


def test_only_one_scheduler_takes_a_task(config, schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    taken = []

    def claim(number: int) -> None:
        if Schedule(config.schedule_path, config.reports_dir).claim(task, LATER):
            taken.append(number)

    in_parallel(claim, 6)
    assert len(taken) == 1


def test_marking_a_report_read_keeps_a_report_added_meanwhile(config, schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    schedule.finish(task, "report written", report=schedule.write_report(task, Run("Any news?", "First"), MONDAY))
    chat = Schedule(config.schedule_path, config.reports_dir)
    original = chat.latest_reports

    def reports_then_a_new_one() -> list:
        reports = original()
        schedule.finish(task, "report written", report=schedule.write_report(task, Run("Any news?", "Second"), LATER))
        return reports

    chat.latest_reports = reports_then_a_new_one
    chat.read_report(1)
    assert [report.read for report in schedule.latest_reports()] == [False, True]


def test_a_lock_held_too_long_is_reported(config, schedule, monkeypatch):
    monkeypatch.setattr("sensai.core.schedule.LOCK_SECONDS", 0)
    with schedule._locked():
        with pytest.raises(ScheduleError, match="locked by another process"):
            Schedule(config.schedule_path, config.reports_dir).add("every 1h", "Any news?", [], MONDAY)


def test_a_failed_change_leaves_the_lock_free(schedule):
    with pytest.raises(ScheduleError, match="No task 1"):
        schedule.remove(1)
    assert schedule.add("every 1h", "Any news?", [], MONDAY)


def test_a_task_taken_by_another_scheduler_runs_once(config, schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    assert Schedule(config.schedule_path, config.reports_dir).claim(task, LATER)
    assert not schedule.claim(task, LATER)


def test_a_task_removed_in_the_meantime_is_not_run(config, schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    Schedule(config.schedule_path, config.reports_dir).remove(1)
    assert not schedule.claim(task, LATER)


def test_finish_keeps_the_report_of_a_task_removed_during_its_run(config, schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    report = schedule.write_report(task, Run("Any news?", "Yes."), LATER)
    Schedule(config.schedule_path, config.reports_dir).remove(1)
    schedule.finish(task, "report written", report=report)
    assert schedule.latest_reports() == [report]


def test_report_file(config, schedule):
    task = schedule.add("daily 08:00", "Any news?", [], MONDAY)
    run = Run("Any news?", "Orion moved [1] [7].", [Excerpts.EXCERPT], ["send: not allowed"])
    report = schedule.write_report(task, run, LATER)
    text = (config.reports_dir / report.file).read_text(encoding="utf-8")
    assert text.startswith("# Any news?\n\nScheduled task (daily 08:00), run on 2026-11-02 at 18:00.\n\nOrion moved")
    assert "- [1] statutes.pdf, p. 1" in text and "[7] " not in text.split("## Sources")[1]
    assert "## Tools that could not run\n\n- send: not allowed" in text


def test_reading_a_report_marks_it_read_the_most_recent_first(schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    schedule.finish(task, "report written", report=schedule.write_report(task, Run("Any news?", "First"), MONDAY))
    schedule.finish(task, "report written", report=schedule.write_report(task, Run("Any news?", "Second"), LATER))
    assert "Second" in schedule.read_report(1)
    assert [report.read for report in schedule.latest_reports()] == [True, False]
    with pytest.raises(ScheduleError, match="No report 3"):
        schedule.read_report(3)


def test_a_deleted_report_is_reported(config, schedule):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    report = schedule.write_report(task, Run("Any news?", "Yes."), LATER)
    schedule.finish(task, "report written", report=report)
    (config.reports_dir / report.file).unlink()
    with pytest.raises(ScheduleError, match="Cannot read the report"):
        schedule.read_report(1)


@pytest.mark.parametrize(("before", "now", "failed_tools", "nothing_new"), [
    ("abc", "abc", [], True),
    ("abc", "def", [], False),
    ("", "abc", [], False),
    ("", "", [], False),
    ("abc", "abc", ["send: not allowed"], False),
])
def test_nothing_new_is_decided_on_the_evidence(schedule, before, now, failed_tools, nothing_new):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    task.evidence = before
    assert Run("q", "NOTHING NEW", failed_tools=failed_tools, evidence=now).nothing_new(task) is nothing_new


def run_twice(config, client, task, sessions, change=lambda: None, **agent):
    """Two runs of the task, with what the scheduler keeps between them. Returns the second one."""
    task.evidence = run_task(Agent(config, client, **agent), task, sessions, MONDAY).evidence
    change()
    return run_task(Agent(config, client, **agent), task, sessions, LATER)


def test_the_same_excerpts_are_nothing_new_whatever_the_model_says(config, schedule, sessions):
    client = FakeClient("Orion moved, a great change!")
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    assert run_twice(config, client, task, sessions, retriever=Excerpts()).nothing_new(task)


def test_the_same_excerpts_in_another_order_are_nothing_new(config, client, schedule, sessions):
    excerpts = Excerpts()
    excerpts.found = [Excerpts.EXCERPT, Excerpts.OTHER]

    def change():
        excerpts.found = [Excerpts.OTHER, Excerpts.EXCERPT]

    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    assert run_twice(config, client, task, sessions, change, retriever=excerpts).nothing_new(task)


def test_a_new_excerpt_is_news_even_if_the_model_says_nothing_changed(config, schedule, sessions):
    client = FakeClient("NOTHING NEW")
    excerpts = Excerpts()

    def change():
        excerpts.found = [Excerpts.EXCERPT, Excerpts.OTHER]

    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    assert not run_twice(config, client, task, sessions, change, retriever=excerpts).nothing_new(task)


def test_an_excerpt_whose_text_changed_is_news(config, client, schedule, sessions):
    excerpts = Excerpts()

    def change():
        excerpts.found = [excerpt(1, "Orion statutes, amended", "statutes.pdf")]

    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    assert not run_twice(config, client, task, sessions, change, retriever=excerpts).nothing_new(task)


def test_a_run_without_evidence_gives_a_report(config, client, schedule, sessions):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    assert not run_twice(config, client, task, sessions).nothing_new(task)


def test_the_same_tool_result_is_nothing_new_and_another_one_is_news(config, client, schedule, sessions):
    answers = iter(["Orion", "Orion", "Kestrel"])

    def look(query: str) -> str:
        """
        Look something up.

        Args:
          query: What to look up
        """
        return next(answers)

    def run(query: str) -> Run:
        client.tool_rounds = [[ToolCall("look", {"query": query})]]
        return run_task(Agent(config, client, tools=ToolRegistry([look])), task, sessions, LATER)

    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    task.evidence = run("a").evidence
    assert run("another query").nothing_new(task)
    assert not run("a").nothing_new(task)


def test_a_tool_that_failed_is_reported_and_gives_a_report(config, client, schedule, sessions):
    def look(query: str) -> str:
        """
        Look something up.

        Args:
          query: What to look up
        """
        raise ToolError("the site is down")

    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    task.evidence = run_task(Agent(config, client, retriever=Excerpts()), task, sessions, MONDAY).evidence
    client.tool_rounds = [[ToolCall("look", {"query": "a"})]]
    run = run_task(Agent(config, client, retriever=Excerpts(), tools=ToolRegistry([look])), task, sessions, LATER)
    assert "the site is down" in run.failed_tools[0] and not run.nothing_new(task)


def test_a_run_with_nothing_new_leaves_the_conversation_as_it_was(config, client, schedule, sessions):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run_twice(config, client, task, sessions, retriever=Excerpts())
    assert sessions.list()[0].turns == 1


def test_the_model_is_told_it_runs_alone_and_only_reports_changes(config, client, schedule, sessions):
    config.system_prompt = "You help journalists."
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run_task(Agent(config, client), task, sessions, LATER)
    system = client.chats[-1][0][0]["content"]
    assert system.startswith("You help journalists.\n\nThis is a scheduled task, run on 2026-11-02 at 18:00.")
    assert "report what changed since the last one" in system


def test_the_search_uses_the_question_only(config, client, schedule, sessions):
    excerpts = Excerpts()
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run = run_task(Agent(config, client, retriever=excerpts), task, sessions, LATER)
    assert excerpts.query == "Any news?" and run.sources == [Excerpts.EXCERPT]


def test_each_run_continues_the_conversation_of_the_task(config, client, schedule, sessions):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run_task(Agent(config, client), task, sessions, MONDAY)
    run_task(Agent(config, client), task, sessions, LATER)
    assert [message["content"] for message in client.chats[-1][0][1:]] == ["Any news?", "Hello there ", "Any news?"]
    assert sessions.list()[0].id == task.id and sessions.list()[0].turns == 2


def test_two_tasks_do_not_share_their_conversation(config, client, schedule, sessions):
    first = schedule.add("every 1h", "First case?", [], MONDAY)
    second = schedule.add("every 1h", "Second case?", [], MONDAY)
    run_task(Agent(config, client), first, sessions, LATER)
    run_task(Agent(config, client), second, sessions, LATER)
    assert "First case?" not in str(client.chats[-1][0])


def test_nothing_is_saved_when_the_sessions_feature_is_off(config, client, schedule, sessions):
    config.features["sessions"] = False
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run_task(Agent(config, client), task, sessions, LATER)
    assert not config.sessions_dir.exists()


def test_the_profile_and_the_persona_are_used(config, client, schedule, sessions):
    agent = Agent(config, client, persona=load_personas(config.personas_dir)["investigator"])
    agent.memory.profile = "- Answer in French."
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run_task(agent, task, sessions, LATER)
    system = client.chats[-1][0][0]["content"]
    assert agent.persona.system_prompt() in system and "- Answer in French." in system


def test_a_critical_tool_is_refused_without_allow(config, client, schedule, sessions):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run = run_task(Agent(config, client, tools=ToolRegistry([send])), task, sessions, LATER)
    assert run.failed_tools == ["send: this scheduled task was not allowed to use it when it was created"]
    tool_message = client.chats[-1][0][-1]
    assert tool_message["role"] == "tool" and "sent Orion" not in tool_message["content"]


def test_a_critical_tool_runs_with_allow(config, client, schedule, sessions):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    task = schedule.add("every 1h", "Any news?", ["send"], MONDAY)
    run = run_task(Agent(config, client, tools=ToolRegistry([send])), task, sessions, LATER)
    assert run.failed_tools == [] and client.chats[-1][0][-1]["content"] == "sent Orion"


def test_allowing_one_tool_does_not_allow_another(config, client, schedule, sessions):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    task = schedule.add("every 1h", "Any news?", ["web_search"], MONDAY)
    run = run_task(Agent(config, client, tools=ToolRegistry([send])), task, sessions, LATER)
    assert run.failed_tools[0].startswith("send: ")


def test_the_cache_and_the_document_are_left_alone(config, client, schedule, sessions):
    agent = Agent(config, client)
    agent.cache, agent.confirm_cache = object(), lambda cached: pytest.fail("the cache was used")
    agent.memory.document = Document()
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    run_task(agent, task, sessions, LATER)
    assert client.edit_requests == [] and agent.memory.turns == []


def guardrails(stage: str, action: str, term: str) -> Guardrails:
    return Guardrails(parse_rules([{"name": "source", "terms": [term], "on": [stage], "action": action}]))


def test_an_answer_goes_through_the_guardrails(config, schedule, sessions):
    client = FakeClient("The source is Claire Benoit.")
    rules = guardrails("output", "redact", "Claire Benoit")
    task = schedule.add("every 1h", "Who?", [], MONDAY)
    run = run_task(Agent(config, client, guardrails=rules), task, sessions, LATER)
    assert "Claire Benoit" not in run.answer


def test_a_question_blocked_since_the_task_was_added_fails(config, client, schedule, sessions):
    rules = guardrails("input", "block", "Claire")
    task = schedule.add("every 1h", "Where is Claire?", [], MONDAY)
    with pytest.raises(ScheduleError, match="blocked by the guardrail source"):
        run_task(Agent(config, client, guardrails=rules), task, sessions, LATER)
    assert client.chats == []


def test_a_question_redacted_since_the_task_was_added_is_redacted_in_the_report(config, client, schedule, sessions):
    rules = guardrails("input", "redact", "Claire")
    task = schedule.add("every 1h", "Where is Claire?", [], MONDAY)
    run = run_task(Agent(config, client, guardrails=rules), task, sessions, LATER)
    assert "Claire" not in run.question


def test_an_empty_answer_fails(config, schedule, sessions):
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    with pytest.raises(ScheduleError, match="empty answer"):
        run_task(Agent(config, FakeClient("")), task, sessions, LATER)


def test_a_compacted_conversation_gives_the_question_too(config, client, schedule, sessions):
    config.keep_recent_turns = 1
    client.usage = (config.num_ctx, 10)
    task = schedule.add("every 1h", "Any news?", [], MONDAY)
    sessions_memory = Agent(config, client).memory
    sessions_memory.turns = [Turn("Earlier", "Answer")]
    sessions.save(task.id, sessions_memory, "m")
    run = run_task(Agent(config, client), task, sessions, LATER)
    assert run.question == "Any news?" and client.summaries
