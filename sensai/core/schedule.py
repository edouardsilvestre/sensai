import hashlib
import json
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from sensai.core.agent import Agent
from sensai.core.approval import Approval
from sensai.core.events import ApprovalEvent, DoneEvent, GuardrailEvent, SourcesEvent, ToolResultEvent
from sensai.core.llm.ollama import ToolCall
from sensai.core.memory.sessions import TASK_PREFIX, SessionStore
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers

if os.name == "nt":
    import msvcrt
else:
    import fcntl

WHEN_FORMS = "every 30m, every 6h, every 2d, daily 08:00 or weekly mon 08:00"
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
MINUTES = {"m": 1, "h": 60, "d": 24 * 60}
MAX_MINUTES = 365 * 24 * 60
NOT_ALLOWED = "this scheduled task was not allowed to use it when it was created"
LOCK_SECONDS = 10
_EVERY = re.compile(r"([0-9]+)([mhd])")

_INSTRUCTIONS = """\
This is a scheduled task, run on {now:%Y-%m-%d at %H:%M}. Nobody is reading along: do not ask questions, give your report.
The earlier reports of this task, if any, are in the conversation: report what changed since the last one."""


class ScheduleError(Exception):
    pass


def _try_lock(file) -> None:
    """Raises OSError when another process holds the lock. The system drops it if that process dies."""
    if os.name == "nt":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(file) -> None:
    if os.name == "nt":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file, fcntl.LOCK_UN)


def _at(text: str, moment: datetime) -> datetime:
    parsed = datetime.strptime(text, "%H:%M")
    return moment.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)


def next_run(when: str, after: datetime) -> datetime:
    """The first time of `when` strictly after `after`. Raises ScheduleError when `when` is not a valid schedule."""
    words = when.lower().split()
    try:
        if len(words) == 2 and words[0] == "every":
            match = _EVERY.fullmatch(words[1])
            minutes = int(match[1]) * MINUTES[match[2]] if match else 0
            if 1 <= minutes <= MAX_MINUTES:
                return after + timedelta(minutes=minutes)
        elif len(words) == 2 and words[0] == "daily":
            moment = _at(words[1], after)
            return moment if moment > after else moment + timedelta(days=1)
        elif len(words) == 3 and words[0] == "weekly" and words[1] in DAYS:
            moment = _at(words[2], after) + timedelta(days=(DAYS.index(words[1]) - after.weekday()) % 7)
            return moment if moment > after else moment + timedelta(days=7)
    except ValueError:
        pass
    raise ScheduleError(f"Invalid schedule '{when}': use {WHEN_FORMS}")


@dataclass
class Task:
    # Also the id of its conversation in the SessionStore.
    id: str
    when: str
    question: str
    next_run: str
    allow: list[str] = field(default_factory=list)
    last_run: str = ""
    status: str = ""
    failed: bool = False
    # The fingerprint of what the last run found, to tell whether the next one found anything else.
    evidence: str = ""


@dataclass
class Report:
    file: str
    question: str
    at: str
    read: bool = False


@dataclass
class Run:
    """What a run of a task produced."""

    question: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    # "tool: reason", for the tools that could not run.
    failed_tools: list[str] = field(default_factory=list)
    evidence: str = ""

    def nothing_new(self, task: Task) -> bool:
        """Decided on what the run found, never on what the model says about it. In doubt, there is a report."""
        return bool(self.evidence) and self.evidence == task.evidence and not self.failed_tools

    def markdown(self, task: Task, now: datetime) -> str:
        lines = [f"# {self.question}", "", f"Scheduled task ({task.when}), run on {now:%Y-%m-%d at %H:%M}.", ""]
        lines += [self.answer.strip(), ""]
        cited = [number for number in sorted(cited_numbers(self.answer)) if 1 <= number <= len(self.sources)]
        if cited:
            lines += ["## Sources", ""] + [f"- [{number}] {self.sources[number - 1].citation}" for number in cited] + [""]
        if self.failed_tools:
            lines += ["## Tools that could not run", ""] + [f"- {failed}" for failed in self.failed_tools] + [""]
        return "\n".join(lines)


def _fingerprint(found: list[str]) -> str:
    # Sorted: the same excerpts in another order are the same evidence.
    return hashlib.sha256("\0".join(sorted(found)).encode()).hexdigest() if found else ""


def _parse(data: object) -> tuple[list[Task], list[Report]]:
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or not isinstance(data.get("reports"), list):
        raise ValueError('expected {"tasks": [...], "reports": [...]}')
    tasks = [Task(**item) for item in data["tasks"]]
    reports = [Report(**item) for item in data["reports"]]
    for task in tasks:
        next_run(task.when, datetime.fromisoformat(task.next_run))
        if not isinstance(task.question, str) or not task.question.strip() or not isinstance(task.allow, list):
            raise ValueError(f"task {task.id} needs a question and a list of allowed tools")
    return tasks, reports


class Schedule:
    """The tasks and the list of reports, in a JSON file that the chat edits while the scheduler runs."""

    def __init__(self, path: Path, reports_dir: Path):
        self.path = path
        self.reports_dir = reports_dir
        self.tasks: list[Task] = []
        self.reports: list[Report] = []
        self.reload()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        # Reading too: on Windows, a file open in another process cannot be replaced.
        path = self.path.with_name(self.path.name + ".lock")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file = path.open("a+b")
        except OSError as e:
            raise ScheduleError(f"Cannot lock the schedule {self.path}: {e}") from None
        with file:
            deadline = time.monotonic() + LOCK_SECONDS
            while True:
                try:
                    _try_lock(file)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ScheduleError(f"The schedule {self.path} is locked by another process") from None
                    time.sleep(0.01)
            try:
                yield
            finally:
                _unlock(file)

    @contextmanager
    def _changing(self) -> Iterator[None]:
        # Another process may have changed the file: every change starts from what it holds now.
        with self._locked():
            self._read()
            yield
            self._save()

    def reload(self) -> None:
        if not self.path.exists():
            self.tasks, self.reports = [], []
            return
        with self._locked():
            self._read()

    def _read(self) -> None:
        if not self.path.exists():
            self.tasks, self.reports = [], []
            return
        try:
            self.tasks, self.reports = _parse(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as e:
            raise ScheduleError(f"Cannot read the schedule {self.path}: {e}") from None
        except (ValueError, TypeError, ScheduleError) as e:
            raise ScheduleError(f"Invalid schedule file {self.path}: {e}") from None

    def _save(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        data = {"tasks": [asdict(task) for task in self.tasks], "reports": [asdict(report) for report in self.reports]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError as e:
            raise ScheduleError(f"Cannot save the schedule {self.path}: {e}") from None

    def _task(self, task_id: str) -> Task | None:
        return next((task for task in self.tasks if task.id == task_id), None)

    def add(self, when: str, question: str, allow: list[str], now: datetime) -> Task:
        first = next_run(when, now)
        with self._changing():
            task_id, number = f"{TASK_PREFIX}{now:%Y%m%d-%H%M%S}", 2
            base = task_id
            while self._task(task_id):
                task_id, number = f"{base}-{number}", number + 1
            task = Task(task_id, " ".join(when.lower().split()), question, first.isoformat(timespec="seconds"), allow)
            self.tasks.append(task)
        return task

    def remove(self, number: int) -> Task:
        with self._changing():
            if not 1 <= number <= len(self.tasks):
                raise ScheduleError(f"No task {number}: the schedule has {len(self.tasks)}")
            return self.tasks.pop(number - 1)

    def due(self, now: datetime) -> list[Task]:
        self.reload()
        return [task for task in self.tasks if datetime.fromisoformat(task.next_run) <= now]

    def claim(self, task: Task, now: datetime) -> bool:
        """Moves the task to its next time before it runs. False when it was removed or taken in the meantime."""
        with self._changing():
            current = self._task(task.id)
            if current is None or datetime.fromisoformat(current.next_run) > now:
                return False
            # From now, not from the missed time: a task missed ten times runs once.
            current.next_run = next_run(current.when, now).isoformat(timespec="seconds")
            current.last_run = now.isoformat(timespec="seconds")
            return True

    def finish(
        self, task: Task, status: str, failed: bool = False, report: Report | None = None, evidence: str | None = None,
    ) -> None:
        with self._changing():
            current = self._task(task.id)
            if current is not None:
                current.status, current.failed = status, failed
                if evidence is not None:
                    current.evidence = evidence
            if report is not None:
                self.reports.append(report)

    def write_report(self, task: Task, run: Run, now: datetime) -> Report:
        name = f"{now:%Y%m%d-%H%M%S}-{task.id}.md"
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            (self.reports_dir / name).write_text(run.markdown(task, now), encoding="utf-8")
        except OSError as e:
            raise ScheduleError(f"Cannot write the report {name}: {e}") from None
        return Report(name, run.question, now.isoformat(timespec="seconds"))

    def latest_reports(self) -> list[Report]:
        self.reload()
        return self.reports[::-1]

    def read_report(self, number: int) -> str:
        """The text of report `number`, the most recent first, which then counts as read."""
        reports = self.latest_reports()
        if not 1 <= number <= len(reports):
            raise ScheduleError(f"No report {number}: there are {len(reports)}")
        report = reports[number - 1]
        try:
            text = (self.reports_dir / report.file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ScheduleError(f"Cannot read the report {report.file}: {e}") from None
        with self._changing():
            for kept in self.reports:
                if kept.file == report.file:
                    kept.read = True
        return text


def task_agent(agent: Agent, task: Task, sessions: SessionStore, now: datetime) -> Agent:
    """An agent for one run: the tools and rules of `agent`, the conversation of the task, and nobody to ask."""

    def approve(call: ToolCall, reason: str) -> Approval:
        return Approval(True) if call.name in task.allow else Approval(False, NOT_ALLOWED)

    runner = Agent(
        agent.config, agent.client, agent.retriever, agent.tools, agent.guardrails, agent.persona,
        approver=approve, examples=agent.examples,
    )
    runner.keeps_document = False
    runner.memory.profile = agent.memory.profile
    if agent.config.features["sessions"] and sessions.exists(task.id):
        sessions.load(task.id, runner.memory)
    parts = [runner.memory.system_prompt, _INSTRUCTIONS.format(now=now)]
    runner.memory.system_prompt = "\n\n".join(part for part in parts if part)
    return runner


def run_task(agent: Agent, task: Task, sessions: SessionStore, now: datetime) -> Run:
    """Runs the task in its own conversation. Raises ScheduleError when the run gives no answer."""
    runner = task_agent(agent, task, sessions, now)
    question, answer, blocked = task.question, "", ""
    sources: list[SearchResult] = []
    failed_tools: list[str] = []
    tool_results: list[str] = []
    for event in runner.run(task.question):
        if isinstance(event, SourcesEvent):
            sources = event.sources
        elif isinstance(event, ApprovalEvent) and not event.approval.approved:
            failed_tools.append(f"{event.call.name}: {NOT_ALLOWED}")
        elif isinstance(event, ToolResultEvent) and event.result.error:
            failed_tools.append(f"{event.call.name}: {event.result.content.removeprefix('Error: ')}")
        elif isinstance(event, ToolResultEvent) and event.result.summary != "refused":
            tool_results.append(f"{event.call.name}\0{event.result.content}")
        elif isinstance(event, GuardrailEvent) and event.hit.stage == "input" and event.hit.action == "block":
            blocked = event.hit.rule
        elif isinstance(event, DoneEvent):
            # The question as the guardrails left it: a rule may have been added since the task was.
            question, answer = runner.memory.turns[-1].user, event.content
    if blocked:
        raise ScheduleError(f"blocked by the guardrail {blocked}")
    if not answer.strip():
        raise ScheduleError("the model gave an empty answer")
    found = [f"{source.citation}\0{source.text}" for source in sources] + tool_results
    run = Run(question, answer, sources, list(dict.fromkeys(failed_tools)), _fingerprint(found))
    # A run that found nothing else leaves no trace, so that the conversation only holds the reports.
    if agent.config.features["sessions"] and not run.nothing_new(task):
        sessions.save(task.id, runner.memory, agent.config.model)
    return run
