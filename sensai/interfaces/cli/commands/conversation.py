import textwrap

from sensai.core.memory.sessions import SessionError
from sensai.core.orchestrator import Review, ReviewError
from sensai.core.rag.models import SearchFilter
from sensai.core.structured import ANSWER_SCHEMA
from sensai.interfaces.cli.answer import chat_turn, name_conversation, show_events
from sensai.interfaces.cli.output import error, plural, shorten
from sensai.interfaces.cli.session import Session, save_session


def sessions_enabled(session: Session) -> bool:
    if not session.config.features["sessions"]:
        print("The sessions feature is off: /feature sessions on to save and resume conversations.")
        return False
    return session.store is not None


def sessions_command(session: Session) -> None:
    if not sessions_enabled(session):
        return
    saved = session.store.list()
    if not saved:
        print(f"No saved conversation yet in {session.store.directory}.")
        return
    for info in saved:
        current = "*" if info.id == session.session_id else " "
        print(f"{current} {info.id}  {info.updated:%Y-%m-%d %H:%M}  {plural(info.turns, 'turn'):>9}  {info.title}")
    print("/resume ID to continue one of them.")


def resume(session: Session, session_id: str) -> bool:
    if session_id == "last":
        session_id = session.store.latest()
        if session_id is None:
            print("No saved conversation yet: this one is saved after your first message.")
            return False
    try:
        turns = session.store.load(session_id, session.agent.memory)
    except SessionError as e:
        error(str(e))
        return False
    session.session_id = session_id
    cleared = forget_conversation(session)
    memory_turns = session.agent.memory.turns
    session.answered = (memory_turns[-1], []) if memory_turns else None
    summary = ", plus a summary of older turns" if session.agent.memory.summary else ""
    print(f"Resumed conversation {session_id}: {plural(turns, 'turn')}{summary}. /context shows them.{cleared}")
    branches = session.agent.memory.branches()
    if len(branches) > 1:
        print(f"On branch {session.agent.memory.branch.number} of {len(branches)}: /branches lists them.")
    return True


def forget_conversation(session: Session) -> str:
    """
    Forgets what belongs to the conversation left, and says which filter it cleared: the filter is shown nowhere, it
    would narrow the searches of the next conversation without the user knowing.
    """
    session.last_sources = []
    session.last_dropped = []
    session.agent.reasoning.clear()
    previous = session.agent.search_filter.describe()
    session.agent.search_filter = SearchFilter()
    return f" Filter {previous} cleared: /filter to set it again." if previous else ""


def resume_command(session: Session, argument: str) -> None:
    if sessions_enabled(session):
        # The current conversation was saved after its last message: switching loses nothing.
        resume(session, argument or "last")


def new_command(session: Session) -> None:
    previous = session.session_id
    session.agent.memory.clear()
    session.session_id = None
    session.agent.session_id = ""
    cleared = forget_conversation(session)
    session.answered = None
    saved = f" Conversation {previous} is saved: /resume {previous} to come back to it." if previous else ""
    print(f"New conversation.{saved}{cleared}")


def history_command(session: Session) -> None:
    memory = session.agent.memory
    if not memory.turns:
        print("No message kept verbatim on this branch." if memory.summary else "No message yet.")
        return
    print(f"Branch {memory.branch.number}:")
    for number, turn in enumerate(memory.turns, start=1):
        print(f"  {number}. {shorten(turn.user)}")
        print(f"     {shorten(turn.assistant)}")
    print("/edit N to try another message instead of one of them.")


def branching_enabled(session: Session) -> bool:
    if not session.config.features["branching"]:
        print("The branching feature is off: /feature branching on to edit past messages and switch branches.")
    return session.config.features["branching"]


def edit_command(session: Session, argument: str) -> None:
    if not branching_enabled(session):
        return
    memory = session.agent.memory
    number, _, text = argument.partition(" ")
    if not number.isdecimal():
        error("Usage: /edit N [TEXT], /history gives the numbers")
        return
    if not 1 <= int(number) <= len(memory.turns):
        error(f"No message {number} on this branch: /history lists them.")
        return
    number = int(number)
    if not text.strip():
        print(f"Message {number}: {memory.turns[number - 1].user}")
        try:
            text = input("New message (empty to cancel): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text.strip():
            return
    previous = memory.branch.number
    memory.fork(number)
    print(f"[branch] Branch {memory.branch.number}, from message {number} of branch {previous}, which stays as it was.")
    forked = memory.turns
    chat_turn(session, text)
    if memory.turns is forked:
        # Blocked by a guardrail, or the model failed: the edit gave nothing to keep.
        memory.discard_branch()
        save_session(session)
        print(f"[branch] Nothing was answered: back on branch {previous}.")


def branches_command(session: Session) -> None:
    if not branching_enabled(session):
        return
    memory = session.agent.memory
    for branch in memory.branches():
        marker = "*" if branch.number == memory.branch.number else " "
        origin = "original" if branch.parent is None else f"from message {branch.turn} of branch {branch.parent}"
        label = f": {shorten(branch.label)}" if branch.label else ""
        print(f"{marker} {branch.number}  {plural(len(branch.turns), 'turn'):>9}  {origin}{label}")
    print("/branch N to continue one of them, /edit N to start another.")


def branch_command(session: Session, argument: str) -> None:
    if not branching_enabled(session):
        return
    memory = session.agent.memory
    if not argument:
        print(f"On branch {memory.branch.number}. /branches lists them.")
    elif not argument.isdecimal():
        error("Usage: /branch N, /branches gives the numbers")
    elif int(argument) == memory.branch.number:
        print(f"Already on branch {argument}.")
    elif not memory.switch(int(argument)):
        error(f"No branch {argument}: /branches lists them.")
    else:
        session.last_sources = []
        session.last_dropped = []
        save_session(session)
        last = f" Last answer: {shorten(memory.turns[-1].assistant)}" if memory.turns else ""
        print(f"On branch {argument}, {plural(len(memory.turns), 'turn')}.{last}")


def context_command(session: Session) -> None:
    memory = session.agent.memory
    usage = memory.last_usage
    used = usage.total_tokens if usage else 0
    print(
        f"Context: {used}/{session.config.num_ctx} tokens used by the last message, "
        f"older turns are condensed from {int(memory.limit)}."
    )
    print(f"History: {plural(len(memory.turns), 'turn')} kept verbatim.")
    if memory.summary:
        size = f" ({memory.summary_usage.completion_tokens} tokens)" if memory.summary_usage else ""
        print(f"Summary of older turns{size}:")
        print(textwrap.indent(memory.summary, "    "))


def review_command(session: Session, argument: str) -> None:
    if not session.config.features["orchestration"]:
        print("The orchestration feature is off: /feature orchestration on to review answers.")
        return
    if not argument:
        error("Usage: /review QUESTION")
        return
    name_conversation(session)
    try:
        review = Review(session.agent, session.personas)
    except ReviewError as e:
        error(f"{e} (in {session.config.personas_dir}).")
        return
    session.last_sources = []
    session.last_dropped = []
    show_events(session, review.run(argument), interrupt=review.interrupt)


def json_command(session: Session, argument: str) -> None:
    if not argument:
        error("Usage: /json QUESTION")
        return
    session.last_sources = []
    session.last_dropped = []
    show_events(session, session.agent.run(argument, ANSWER_SCHEMA), structured=True)
