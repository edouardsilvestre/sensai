import argparse
import inspect
import json
import shlex
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from sensai.core.agent import Agent, EmptyInputError
from sensai.core.config import Config, ConfigError, load_config
from sensai.core.events import (
    CompactionEvent,
    CompactionFailedEvent,
    CompactionStartedEvent,
    DoneEvent,
    SourcesEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.llm.ollama import LLMError, OllamaClient, ToolCall
from sensai.core.memory.profile import Profile, ProfileError
from sensai.core.memory.sessions import SessionError, SessionStore
from sensai.core.rag.index import DocumentIndex, IngestProgress, IngestReport
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers
from sensai.core.rag.store import StoreError, VectorStore
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.registry import ToolRegistry

PROMPT = "\n> "
EXIT_COMMANDS = {"/exit", "/quit"}
HELP = """\
Commands:
  /ingest PATH...         index files or folders so that answers can draw on them
  /sources                show the excerpts retrieved for the last message
  /rag [on|off]           turn document search on or off
  /tools [on|off]         list the tools the model can call, or turn them on or off
  /context                show the context usage and the summary of older turns
  /sessions               list the saved conversations
  /resume [ID]            continue a saved conversation (the last one without ID)
  /new                    start a new conversation, the current one stays saved
  /profile                show the instructions the model gets about you
  /profile add TEXT       add an instruction to your profile
  /profile remove N       remove instruction N from your profile
  /exit, /quit            leave (or press Ctrl+D)"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai CLI chatbot")
    parser.add_argument("-e", "--env-file", dest="env_path", type=Path, help="Path to a .env file (default: ./.env)")
    parser.add_argument(
        "-r", "--resume", nargs="?", const="last", metavar="ID",
        help="continue a saved conversation (the last one without ID)",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser("chat", help="chat with the model (default)")
    ingest_parser = commands.add_parser("ingest", help="index documents for search")
    ingest_parser.add_argument("paths", nargs="+", type=Path, metavar="PATH", help="files or folders to index")
    ingest_parser.add_argument(
        "--rebuild", action="store_true", help="empty the index first (needed after changing the embedding model)"
    )
    commands.add_parser("status", help="show what the document index contains")
    return parser.parse_args(argv)


def error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr)


def warning(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr)


def open_index(config: Config, client: OllamaClient, rebuild: bool = False) -> DocumentIndex:
    client.ensure_model(config.embed_model)
    return DocumentIndex.open(config, client, rebuild=rebuild)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def print_progress(progress: IngestProgress) -> None:
    if progress.status == "failed":
        detail = f"failed: {progress.error}"
    elif progress.status == "unchanged":
        detail = "unchanged"
    else:
        detail = f"{progress.status} ({plural(progress.chunks, 'chunk')})"
    width = len(str(progress.total))
    print(f"[{progress.position:>{width}}/{progress.total}] {progress.name}: {detail}")


def print_report(report: IngestReport, index: DocumentIndex) -> None:
    print(
        f"Done in {report.seconds:.1f}s: {report.added} added, {report.updated} updated, "
        f"{report.unchanged} unchanged, {report.removed} removed, {len(report.failed)} failed."
    )
    if report.unsupported:
        print(f"{plural(report.unsupported, 'file')} skipped: unsupported type.")
    for name, reason in report.failed:
        warning(f"{name}: {reason}")
    stats = index.stats()
    print(f"Index: {plural(stats.documents, 'document')}, {plural(stats.chunks, 'chunk')}.")


def ingest(index: DocumentIndex, paths: list[Path]) -> bool:
    try:
        report = index.ingest(paths, on_progress=print_progress)
    except LLMError as e:
        error(f"Ingestion aborted: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[interrupted] Files indexed so far are kept: run the same command again to resume.")
        return False
    print_report(report, index)
    return True


def run_ingest(config: Config, paths: list[Path], rebuild: bool) -> int:
    try:
        index = open_index(config, OllamaClient(config.host), rebuild=rebuild)
    except (LLMError, StoreError) as e:
        error(str(e))
        return 1
    try:
        print(f"Indexing into {config.index_path} with {config.embed_model}")
        return 0 if ingest(index, paths) else 1
    finally:
        index.close()


def run_status(config: Config) -> int:
    if not config.index_path.exists():
        print(f"No index at {config.index_path}. Create one with: python -m sensai ingest PATH")
        return 0
    try:
        store = VectorStore(config.index_path)
    except StoreError as e:
        error(str(e))
        return 1
    try:
        stats = store.stats()
    finally:
        store.close()
    size = config.index_path.stat().st_size / 1_000_000
    print(f"Index: {config.index_path} ({size:.1f} MB)")
    dimensions = f" ({stats.dimensions} dimensions)" if stats.dimensions else ""
    print(f"Embedding model: {stats.embed_model or '-'}{dimensions}")
    print(f"Documents: {stats.documents}, chunks: {stats.chunks}")
    if stats.embed_model and stats.embed_model != config.embed_model:
        warning(f"SENSAI_EMBED_MODEL is '{config.embed_model}': search is disabled until it matches the index.")
    return 0


@dataclass
class Session:
    config: Config
    agent: Agent
    index: DocumentIndex | None = None
    last_sources: list[SearchResult] = field(default_factory=list)
    store: SessionStore | None = None
    # Given at the first save, so that a conversation without any message is never saved.
    session_id: str | None = None
    profile: Profile | None = None


def print_sources(sources: list[SearchResult], answer: str) -> None:
    cited = cited_numbers(answer)
    valid = [number for number in cited if 1 <= number <= len(sources)]
    if valid:
        print("\nSources:")
        for number in sorted(valid):
            print(f"  [{number}] {sources[number - 1].citation}")
    else:
        print(f"\n(no excerpt cited; /sources shows the {len(sources)} excerpts retrieved)")
    unknown = [number for number in cited if number not in valid]
    if unknown:
        warning("the answer cites excerpts that do not exist: " + " ".join(f"[{n}]" for n in unknown))


def show_excerpts(sources: list[SearchResult]) -> None:
    if not sources:
        print("No excerpts were retrieved for the last message.")
        return
    for number, source in enumerate(sources, start=1):
        print(f"\n[{number}] {source.citation} (similarity {source.score:.2f})")
        print(textwrap.indent(source.text, "    "))


def describe_call(call: ToolCall) -> str:
    return ", ".join(f"{name}={json.dumps(value, ensure_ascii=False)}" for name, value in call.arguments.items())


def chat_turn(session: Session, line: str) -> None:
    sources: list[SearchResult] = []
    if line.strip():
        session.last_sources = []
    mid_line = False
    try:
        for event in session.agent.run(line):
            if isinstance(event, SourcesEvent):
                sources = session.last_sources = event.sources
                documents = len({source.source for source in sources})
                print(f"[search] {len(sources)} excerpts from {documents} documents")
            elif isinstance(event, TokenEvent):
                print(event.text, end="", flush=True)
                mid_line = not event.text.endswith("\n")
            elif isinstance(event, ToolCallEvent):
                if mid_line:
                    print()
                    mid_line = False
                print(f"[{event.call.name}] {describe_call(event.call)}")
            elif isinstance(event, ToolResultEvent):
                if event.result.error:
                    warning(f"{event.call.name} failed: {event.result.content.removeprefix('Error: ')}")
                else:
                    print(f"[{event.call.name}] {event.result.summary or 'done'}")
            elif isinstance(event, DoneEvent):
                print()
                if sources:
                    print_sources(sources, event.content)
            elif isinstance(event, CompactionStartedEvent):
                print(f"[memory] context at {event.tokens}/{session.config.num_ctx} tokens, condensing the oldest turns...")
            elif isinstance(event, CompactionEvent):
                print(f"[memory] {plural(event.turns, 'turn')} condensed into the summary.")
            elif isinstance(event, CompactionFailedEvent):
                warning(f"Could not condense the history, retrying after the next message: {event.message}")
    except EmptyInputError:
        error("Empty message, please type something.")
    except (LLMError, StoreError) as e:
        print()
        error(str(e))
    except KeyboardInterrupt:
        print("\n[interrupted]")
    save_session(session)


def save_session(session: Session) -> None:
    memory = session.agent.memory
    if session.store is None or not session.config.features["sessions"] or not (memory.turns or memory.summary):
        return
    if session.session_id is None:
        session.session_id = session.store.new_id()
    try:
        session.store.save(session.session_id, memory, session.config.model)
    except SessionError as e:
        warning(f"The conversation is not saved: {e}")


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
    session.last_sources = []
    summary = ", plus a summary of older turns" if session.agent.memory.summary else ""
    print(f"Resumed conversation {session_id}: {plural(turns, 'turn')}{summary}. /context shows them.")
    return True


def resume_command(session: Session, argument: str) -> None:
    if sessions_enabled(session):
        # The current conversation was saved after its last message: switching loses nothing.
        resume(session, argument or "last")


def new_command(session: Session) -> None:
    memory = session.agent.memory
    previous = session.session_id
    memory.summary, memory.turns, memory.last_usage = "", [], None
    session.session_id = None
    session.last_sources = []
    saved = f" Conversation {previous} is saved: /resume {previous} to come back to it." if previous else ""
    print(f"New conversation.{saved}")


def profile_command(session: Session, argument: str) -> None:
    profile = session.profile
    if profile is None:
        error("The profile could not be read at startup: fix the file and restart Sensai.")
        return
    action, _, value = argument.partition(" ")
    try:
        if action == "add":
            profile.add(value)
            print(f"Added as instruction {len(profile.entries)}.")
        elif action == "remove":
            if not value.strip().isdigit():
                error("Usage: /profile remove N")
                return
            print(f"Removed: {profile.remove(int(value))}")
        elif action:
            error("Usage: /profile [add TEXT | remove N]")
            return
    except ProfileError as e:
        error(str(e))
        return
    session.agent.memory.profile = profile.text()
    if not action:
        show_profile(session)
    elif not session.config.features["profile"]:
        print("The profile feature is off: /feature profile on to send it to the model.")


def show_profile(session: Session) -> None:
    profile = session.profile
    if not profile.entries:
        print(f"Your profile is empty ({profile.path}). Add what the model should always know, for example:")
        print("  /profile add I work on the Alvor Consulting case for Le Monde; answer in French.")
        return
    state = "" if session.config.features["profile"] else " (not sent: the profile feature is off)"
    print(f"Profile ({profile.path}){state}:")
    for number, entry in enumerate(profile.entries, start=1):
        print(f"  {number}. {entry}")


def ingest_command(session: Session, argument: str) -> None:
    try:
        paths = [Path(path) for path in shlex.split(argument)]
    except ValueError as e:
        error(f"Cannot read the paths: {e}")
        return
    if not paths:
        error("Usage: /ingest PATH...")
        return
    if session.index is None:
        try:
            session.index = open_index(session.config, session.agent.client)
        except (LLMError, StoreError) as e:
            error(str(e))
            return
        session.agent.retriever = session.index
    ingest(session.index, paths)


def rag_command(session: Session, argument: str) -> None:
    if argument in ("on", "off"):
        session.agent.rag_enabled = argument == "on"
    elif argument:
        error("Usage: /rag [on|off]")
        return
    state = "on" if session.agent.rag_enabled else "off"
    if session.index is None:
        print(f"Document search is {state}, but there is no index yet: use /ingest PATH.")
    else:
        print(f"Document search is {state}.")


def tools_command(session: Session, argument: str) -> None:
    agent = session.agent
    if argument in ("on", "off"):
        agent.tools_enabled = argument == "on"
    elif argument:
        error("Usage: /tools [on|off]")
        return
    if not agent.tools:
        if not session.config.tools:
            print("Tools are disabled by SENSAI_TOOLS=false.")
        else:
            print("No tool is enabled. Set SENSAI_WEB_SEARCH=true to let the model search the web.")
        return
    print(f"Tools are {'on' if agent.tools_enabled else 'off'}:")
    for tool in agent.tools.tools():
        description = " ".join((inspect.getdoc(tool) or "").split("\n\n")[0].split())
        print(f"  {tool.__name__}: {description}")


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


def handle_command(session: Session, line: str) -> bool:
    name, _, argument = line.strip().partition(" ")
    argument = argument.strip()
    if name in EXIT_COMMANDS:
        return False
    if name == "/help":
        print(HELP)
    elif name == "/sources":
        show_excerpts(session.last_sources)
    elif name == "/rag":
        rag_command(session, argument)
    elif name == "/tools":
        tools_command(session, argument)
    elif name == "/ingest":
        ingest_command(session, argument)
    elif name == "/context":
        context_command(session)
    elif name == "/sessions":
        sessions_command(session)
    elif name == "/resume":
        resume_command(session, argument)
    elif name == "/new":
        new_command(session)
    elif name == "/profile":
        profile_command(session, argument)
    else:
        error(f"Unknown command {name}. Type /help for the list.")
    return True


def load_existing_index(config: Config, client: OllamaClient) -> DocumentIndex | None:
    if not config.index_path.exists():
        return None
    try:
        return open_index(config, client)
    except (LLMError, StoreError) as e:
        warning(f"Document search disabled: {e}")
        return None


def load_tools(config: Config, client: OllamaClient) -> ToolRegistry:
    tools = build_registry(config, client)
    if tools and not client.can_call_tools(config.model):
        warning(f"Model '{config.model}' cannot call tools, they are disabled. Pick a model like qwen3 or llama3.1.")
        return ToolRegistry()
    return tools


def load_memory(session: Session, resume_id: str | None) -> None:
    """Loads the profile, then the conversation to resume. Neither can stop the chat from starting."""
    config = session.config
    try:
        session.profile = Profile(config.profile_path)
    except ProfileError as e:
        warning(f"{e}. The profile is not sent to the model.")
    else:
        session.agent.memory.profile = session.profile.text()
        if session.profile.entries and config.features["profile"]:
            print(f"Profile: {plural(len(session.profile.entries), 'instruction')} sent to the model. /profile to see them.")
    session.store = SessionStore(config.sessions_dir)
    if resume_id is None:
        return
    if config.features["sessions"]:
        resume(session, resume_id)
    else:
        warning("--resume is ignored: the sessions feature is off.")


def run_chat(config: Config, resume_id: str | None = None) -> int:
    agent = Agent(config)
    try:
        agent.client.ensure_model(config.model)
        agent.tools = load_tools(config, agent.client)
    except LLMError as e:
        error(str(e))
        return 1

    session = Session(config, agent, index=load_existing_index(config, agent.client))
    agent.retriever = session.index
    print(f"Sensai - model: {config.model}. Type /help for commands, /exit or Ctrl+D to quit.")
    if session.index is not None:
        stats = session.index.stats()
        print(f"Document search: {stats.documents} documents indexed ({stats.chunks} chunks).")
    else:
        print("No documents indexed yet: /ingest PATH to search your files.")
    if agent.tools:
        print(f"Tools: {', '.join(agent.tools.names())}. /tools to see what they do.")
    load_memory(session, resume_id)

    try:
        while True:
            try:
                line = input(PROMPT)
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.env_path)
    except ConfigError as e:
        error(str(e))
        return 1

    if args.command == "ingest":
        return run_ingest(config, args.paths, args.rebuild)
    if args.command == "status":
        return run_status(config)
    return run_chat(config, args.resume)


if __name__ == "__main__":
    sys.exit(main())
