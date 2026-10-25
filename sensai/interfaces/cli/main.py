import argparse
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
)
from sensai.core.llm.ollama import LLMError, OllamaClient
from sensai.core.rag.index import DocumentIndex, IngestProgress, IngestReport
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers
from sensai.core.rag.store import StoreError, VectorStore

PROMPT = "\n> "
EXIT_COMMANDS = {"/exit", "/quit"}
HELP = """\
Commands:
  /ingest PATH...   index files or folders so that answers can draw on them
  /sources          show the excerpts retrieved for the last message
  /rag [on|off]     turn document search on or off
  /context          show the context usage and the summary of older turns
  /exit, /quit      leave (or press Ctrl+D)"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai CLI chatbot")
    parser.add_argument("-e", "--env-file", dest="env_path", type=Path, help="Path to a .env file (default: ./.env)")
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


def chat_turn(session: Session, line: str) -> None:
    sources: list[SearchResult] = []
    if line.strip():
        session.last_sources = []
    try:
        for event in session.agent.run(line):
            if isinstance(event, SourcesEvent):
                sources = session.last_sources = event.sources
                documents = len({source.source for source in sources})
                print(f"[search] {len(sources)} excerpts from {documents} documents")
            elif isinstance(event, TokenEvent):
                print(event.text, end="", flush=True)
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
    elif name == "/ingest":
        ingest_command(session, argument)
    elif name == "/context":
        context_command(session)
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


def run_chat(config: Config) -> int:
    agent = Agent(config)
    try:
        agent.client.ensure_model(config.model)
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
    return run_chat(config)


if __name__ == "__main__":
    sys.exit(main())
