from pathlib import Path

from sensai.core.export.report import FORMATS, ExportError, Report, conversation_report, document_report, render
from sensai.core.memory.sessions import session_data
from sensai.interfaces.cli.output import error
from sensai.interfaces.cli.session import Session, save_session, saved_fields


def document_command(session: Session, argument: str) -> None:
    memory = session.agent.memory
    action, _, value = argument.partition(" ")
    if action == "undo":
        if memory.document.undo():
            print("The changes of the last answer are undone.")
            save_session(session)
        else:
            print("Nothing to undo.")
    elif action == "export":
        path = Path(value.strip().strip('"'))
        if not value.strip():
            error("Usage: /document export PATH")
        elif write_report(report_of(session, "document"), path):
            print(f"Document written to {path}.")
    elif action:
        error("Usage: /document [undo | export PATH]")
    else:
        show_document(session)


def report_of(session: Session, kind: str) -> Report:
    memory = session.agent.memory
    if kind == "document":
        return document_report(memory.document)
    data = session_data(session.session_id, memory, session.config.model, **saved_fields(session))
    return conversation_report(memory, session.config.model, data)


def write_report(report: Report, path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        error(f"Cannot write {path}: the name must end with {', '.join(FORMATS)}, which says the format.")
        return False
    try:
        content = render(report, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except ExportError as e:
        error(str(e))
        return False
    except OSError as e:
        error(f"Cannot write {path}: {e}")
        return False
    return True


def export_command(session: Session, argument: str) -> None:
    memory = session.agent.memory
    if not argument:
        error(f"Usage: /export PATH, with PATH ending in {', '.join(FORMATS)}")
    elif not (memory.turns or memory.summary):
        print("No message to export yet.")
    else:
        path = Path(argument.strip('"'))
        if write_report(report_of(session, "conversation"), path):
            print(f"Conversation written to {path}.")
            branches = len(memory.branches())
            if branches > 1 and path.suffix.lower() != ".json":
                print(f"It holds branch {memory.branch.number} only, out of {branches}: a .json file holds all of them.")


def show_document(session: Session) -> None:
    document = session.agent.memory.document
    if not session.config.features["document"]:
        print("The document feature is off: /feature document on to update it after every answer.")
    if document.is_empty:
        print(f"The document is empty. It fills in as the conversation establishes facts: {', '.join(document.names())}.")
        return
    print(document.markdown().rstrip())
