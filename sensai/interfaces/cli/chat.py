from sensai.core.config import Config
from sensai.interfaces.cli.answer import chat_turn
from sensai.interfaces.cli.commands.dispatch import handle_command
from sensai.interfaces.cli.output import plural, warning
from sensai.interfaces.cli.session import Session, ask_approval, confirm_cached
from sensai.interfaces.cli.startup import load_cache, load_log, load_memory, load_schedule, start_session


def prompt(session: Session) -> str:
    persona = session.agent.persona
    return f"\n[{persona.id}] > " if persona else "\n> "


def run_chat(config: Config, resume_id: str | None = None) -> int:
    session = start_session(config)
    if session is None:
        return 1
    agent = session.agent
    if config.features["cache"]:
        session.cache = agent.cache = load_cache(config, agent.client)
    if config.features["logging"]:
        session.log = agent.log = load_log(config)
    agent.approver = lambda call, reason: ask_approval(session, call, reason)
    agent.confirm_cache = confirm_cached
    print(f"Sensai - model: {config.model}. Type /help for commands, /exit or Ctrl+D to quit.")
    disabled = [name for name, enabled in config.features.items() if not enabled]
    if disabled:
        print(f"Disabled features: {', '.join(disabled)}. /features to list them.")
    if config.features["rag"]:
        if session.index is not None:
            stats = session.index.stats()
            print(f"Document search: {stats.documents} documents indexed ({stats.chunks} chunks).")
        else:
            print("No documents indexed yet: /ingest PATH to search your files.")
    if session.cache is not None:
        print(f"Cache: {plural(session.cache.stats().answers, 'answer')} kept. /cache to see them.")
    if session.examples is not None and session.examples.examples and config.features["examples"]:
        kept = plural(len(session.examples.examples), "answer")
        print(f"Examples: {kept} kept as examples. /examples to list them.")
    if agent.tools:
        print(f"Tools: {', '.join(agent.tools.names())}. /tools to see what they do.")
    if agent.persona:
        version = f" (prompt version {agent.prompt_version})" if isinstance(agent.prompt_version, int) else ""
        print(f"Persona: {agent.persona.name}{version}. /personas to list them, /persona ID to switch.")
    elif session.personas:
        print("No persona: /personas to list them, /persona ID to pick one.")
    load_memory(session, resume_id)
    if config.features["schedule"]:
        session.schedule = load_schedule(config)
        print_schedule_news(session)

    try:
        while True:
            try:
                line = input(prompt(session))
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if line.strip().startswith("/"):
                if not handle_command(session, line):
                    return 0
                continue
            chat_turn(session, line)
    finally:
        if session.index is not None:
            session.index.close()
        if session.cache is not None:
            session.cache.close()
        if session.log is not None:
            session.log.close()


def print_schedule_news(session: Session) -> None:
    schedule = session.schedule
    if schedule is None:
        return
    unread = sum(not report.read for report in schedule.reports)
    failed = sum(task.failed for task in schedule.tasks)
    if unread:
        print(f"[schedule] {plural(unread, 'new report')}. /reports to read them.")
    if failed:
        warning(f"{plural(failed, 'scheduled task')} failed on the last run: /schedule to see why.")
