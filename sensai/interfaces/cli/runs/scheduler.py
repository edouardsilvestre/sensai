import time
from datetime import datetime

from sensai.core.config import Config
from sensai.core.schedule import ScheduleError, run_task
from sensai.interfaces.cli.output import error, plural, warning
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import load_memory, load_schedule, start_session

POLL_SECONDS = 15


def run_due_tasks(session: Session) -> bool:
    """Runs the tasks that are due, one after the other. False when one failed: the next ones still run."""
    schedule, succeeded = session.schedule, True
    for task in schedule.due(datetime.now()):
        now = datetime.now()
        if not schedule.claim(task, now):
            continue
        print(f"[{now:%H:%M}] {task.question}", flush=True)
        try:
            run = run_task(session.agent, task, session.store, now)
            report = None if run.nothing_new(task) else schedule.write_report(task, run, now)
        except Exception as e:
            # Whatever a tool raises, nobody is there to restart the scheduler: one task must not stop the others.
            reason = str(e) or type(e).__name__
            warning(f"The task failed, it runs again at its next time: {reason}")
            schedule.finish(task, f"failed: {reason}", failed=True)
            succeeded = False
            continue
        except KeyboardInterrupt:
            schedule.finish(task, "failed: interrupted", failed=True)
            raise
        for failed in run.failed_tools:
            warning(f"A tool could not run, {failed}")
        if report is None:
            schedule.finish(task, "nothing new", evidence=run.evidence)
            print("Nothing new.", flush=True)
        else:
            schedule.finish(task, "report written", report=report, evidence=run.evidence)
            print(f"New report: {schedule.reports_dir / report.file}", flush=True)
    return succeeded


def watch_schedule(session: Session, once: bool) -> int:
    try:
        while True:
            try:
                succeeded = run_due_tasks(session)
            except ScheduleError as e:
                error(str(e))
                succeeded = False
            if once:
                return 0 if succeeded else 1
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[interrupted] The scheduler stopped: late tasks run when it starts again.")
        return 0


def run_schedule(config: Config, once: bool) -> int:
    if not config.features["schedule"]:
        error("The schedule feature is off: SENSAI_FEATURE_SCHEDULE=true to run the tasks.")
        return 1
    session = start_session(config, asker=None)
    if session is None:
        return 1
    load_memory(session, None)
    session.schedule = load_schedule(config)
    try:
        if session.schedule is None:
            return 1
        tasks = plural(len(session.schedule.tasks), "task")
        print(f"Sensai scheduler - model: {config.model}, {tasks}." + ("" if once else " Ctrl+C to stop."), flush=True)
        return watch_schedule(session, once)
    finally:
        if session.index is not None:
            session.index.close()
