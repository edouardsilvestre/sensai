import re
from datetime import datetime

from sensai.core.llm.ollama import LLMError
from sensai.core.schedule import WHEN_FORMS, ScheduleError
from sensai.interfaces.cli.answer import print_guardrail
from sensai.interfaces.cli.output import error
from sensai.interfaces.cli.session import Session

SCHEDULE_ADD = re.compile(r"(every \S+|daily \S+|weekly \S+ \S+)(?: --allow (\S+))? (.+)", re.IGNORECASE)
SCHEDULE_USAGE = f"Usage: /schedule add WHEN [--allow TOOL,TOOL] TASK, or /schedule remove N. WHEN is {WHEN_FORMS}"


def run_time(value: str) -> str:
    return f"{datetime.fromisoformat(value):%Y-%m-%d %H:%M}"


def add_task(session: Session, text: str) -> None:
    match = SCHEDULE_ADD.fullmatch(" ".join(text.split()))
    if match is None or match[3].startswith("--"):
        error(SCHEDULE_USAGE)
        return
    allow = [name for name in (match[2] or "").split(",") if name]
    unknown = sorted(set(allow) - set(session.agent.tools.names()))
    if unknown:
        error(f"Cannot allow {', '.join(unknown)}: /tools lists the tools enabled.")
        return
    checked = session.agent.active_guardrails.check_input(match[3])
    for hit in checked.hits:
        print_guardrail(hit)
    if checked.blocked:
        print_guardrail(checked.blocked)
        return
    task = session.schedule.add(match[1], checked.text, allow, datetime.now())
    print(f"Task {len(session.schedule.tasks)} added, first run on {run_time(task.next_run)}.")
    print("It runs while python -m sensai schedule is running.")


def show_schedule(session: Session) -> None:
    session.schedule.reload()
    tasks = session.schedule.tasks
    if not tasks:
        print("No scheduled task. For example: /schedule add daily 08:00 What is new about Orion Delta?")
        return
    now = datetime.now()
    late = False
    print("Scheduled tasks:")
    for number, task in enumerate(tasks, start=1):
        overdue = datetime.fromisoformat(task.next_run) <= now
        late = late or overdue
        last = f"last run {run_time(task.last_run)}: {task.status or 'running'}" if task.last_run else "not run yet"
        allowed = f", allowed: {', '.join(task.allow)}" if task.allow else ""
        print(f"  {number}. {task.question}")
        print(f"     {task.when}, next run {run_time(task.next_run)}{' (late)' if overdue else ''}, {last}{allowed}")
    if late:
        print("Late tasks run as soon as python -m sensai schedule is started.")


def schedule_ready(session: Session) -> bool:
    if not session.config.features["schedule"]:
        print("The schedule feature is off: /feature schedule on to run tasks on their own.")
        return False
    if session.schedule is None:
        print("The schedule could not be read: fix the file, then /feature schedule on.")
        return False
    return True


def schedule_command(session: Session, argument: str) -> None:
    if not schedule_ready(session):
        return
    action, _, value = argument.partition(" ")
    value = value.strip()
    try:
        if action == "add":
            add_task(session, value)
        elif action == "remove" and value.isdecimal():
            print(f"Removed: {session.schedule.remove(int(value)).question}")
        elif action:
            error(SCHEDULE_USAGE)
        else:
            show_schedule(session)
    except (ScheduleError, LLMError) as e:
        error(str(e))


def reports_command(session: Session, argument: str) -> None:
    if not schedule_ready(session):
        return
    try:
        if argument:
            if not argument.isdecimal():
                error("Usage: /reports [N]")
                return
            print(session.schedule.read_report(int(argument)).rstrip())
            return
        reports = session.schedule.latest_reports()
    except ScheduleError as e:
        error(str(e))
        return
    if not reports:
        print("No report yet: a report is written when a scheduled task finds something new.")
        return
    print("Reports (* not read yet):")
    for number, report in enumerate(reports, start=1):
        print(f"  {' ' if report.read else '*'} {number}. {run_time(report.at)}  {report.question}")
    print("/reports N to read one.")
