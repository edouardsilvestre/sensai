import argparse
import inspect
import json
import shlex
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from sensai.core.agent import Agent, EmptyInputError
from sensai.core.config import API_KEY_HELP, FEATURES, Config, ConfigError, load_config
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
from sensai.core.persona import Persona, PersonaError, load_personas
from sensai.core.rag.index import DocumentIndex, IngestProgress, IngestReport
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers
from sensai.core.rag.store import StoreError, VectorStore
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.registry import ToolRegistry

EXIT_COMMANDS = {"/exit", "/quit"}
HELP = """\
Commands:
  /ingest PATH...         index files or folders so that answers can draw on them
  /sources                show the excerpts retrieved for the last message
  /tools                  list the tools the model can call
  /context                show the context usage and the summary of older turns
  /features               list the features and whether they are on
  /feature NAME on|off    turn a feature on or off for this session
  /personas               list the personas
  /persona [ID]           show the active persona, or switch to another one (the history is kept)
  /exit, /quit            leave (or press Ctrl+D)"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai CLI chatbot")
    parser.add_argument("-e", "--env-file", dest="env_path", type=Path, help="Path to a .env file (default: ./.env)")
    parser.add_argument("-p", "--persona", help="persona to start with (overrides SENSAI_PERSONA)")
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


def feature_states(config: Config) -> str:
    return ", ".join(f"{name} {'on' if enabled else 'off'}" for name, enabled in config.features.items())


def run_status(config: Config) -> int:
    print(f"Features: {feature_states(config)}")
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
    personas: dict[str, Persona] = field(default_factory=dict)


def prompt(session: Session) -> str:
    persona = session.agent.persona
    return f"\n[{persona.id}] > " if persona else "\n> "


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
    # Multi-line arguments, like code, are shown as indented blocks instead of one escaped line.
    inline = []
    blocks = []
    for name, value in call.arguments.items():
        if isinstance(value, str) and "\n" in value.strip():
            blocks.append(f"\n  {name}:\n" + textwrap.indent(value.strip("\n"), "    "))
        else:
            inline.append(f"{name}={json.dumps(value, ensure_ascii=False)}")
    return f"[{call.name}] {', '.join(inline)}".rstrip() + "".join(blocks)


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
                print(describe_call(event.call))
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
    if ingest(session.index, paths) and not session.config.features["rag"]:
        print("The rag feature is off: /feature rag on to answer from these documents.")


def tools_command(session: Session) -> None:
    agent = session.agent
    if not session.config.features["tools"]:
        print("The tools feature is off: /feature tools on to let the model call tools.")
        return
    if not agent.tools:
        print("No tool is enabled: /feature web_search on to let the model search the web.")
        return
    print("Tools:")
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


def list_features(session: Session) -> None:
    width = max(len(name) for name in FEATURES)
    for name, feature in FEATURES.items():
        state = "on" if session.config.features[name] else "off"
        print(f"  {name:<{width}}  {state:<3}  {feature.description}")


def feature_command(session: Session, argument: str) -> None:
    name, _, value = argument.lower().partition(" ")
    value = value.strip()
    if name not in FEATURES or value not in ("on", "off"):
        error(f"Usage: /feature NAME on|off, with NAME among: {', '.join(FEATURES)}")
        return
    config = session.config
    if name == "web_search" and value == "on" and not config.ollama_api_key:
        error(f"web_search needs OLLAMA_API_KEY in .env: {API_KEY_HELP}")
        return
    config.features[name] = value == "on"
    print(f"{name} is {value}.")
    if name in ("tools", "web_search") and config.features["tools"]:
        # The registry only holds the tools enabled when it was built: build it again to add or remove one.
        reload_tools(session)
    if name == "rag" and value == "on" and session.index is None:
        if config.index_path.exists():
            session.index = load_existing_index(config, session.agent.client)
            session.agent.retriever = session.index
        else:
            print("No documents indexed yet: /ingest PATH to search your files.")


def reload_tools(session: Session) -> None:
    try:
        session.agent.tools = load_tools(session.config, session.agent.client)
    except LLMError as e:
        error(str(e))
        return
    if session.agent.tools:
        print(f"Tools: {', '.join(session.agent.tools.names())}. /tools to see what they do.")


def list_personas(session: Session) -> None:
    if not session.personas:
        print(f"No persona in {session.config.personas_dir}.")
        return
    active = session.agent.persona
    width = max(len(persona_id) for persona_id in session.personas)
    for persona in session.personas.values():
        marker = "*" if active and active.id == persona.id else " "
        print(f" {marker} {persona.id:<{width}}  {persona.name}: {persona.role}")


def unknown_persona(personas: dict[str, Persona], persona_id: str) -> str:
    return f"Unknown persona '{persona_id}'. Available: {', '.join(personas) or 'none'}"


def persona_command(session: Session, argument: str) -> None:
    agent = session.agent
    if not argument:
        if agent.persona:
            print(f"Active persona: {agent.persona.id} ({agent.persona.name}). /personas to list them.")
        else:
            print("No active persona. /personas to list them.")
        return
    persona = session.personas.get(argument)
    if persona is None:
        error(unknown_persona(session.personas, argument))
        return
    agent.set_persona(persona)
    print(f"Switched to {persona.name}. The conversation history is kept.")


def handle_command(session: Session, line: str) -> bool:
    name, _, argument = line.strip().partition(" ")
    argument = argument.strip()
    if name in EXIT_COMMANDS:
        return False
    if name == "/help":
        print(HELP)
    elif name == "/sources":
        show_excerpts(session.last_sources)
    elif name == "/tools":
        tools_command(session)
    elif name == "/ingest":
        ingest_command(session, argument)
    elif name == "/context":
        context_command(session)
    elif name == "/features":
        list_features(session)
    elif name == "/feature":
        feature_command(session, argument)
    elif name == "/personas":
        list_personas(session)
    elif name == "/persona":
        persona_command(session, argument)
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
    for message in tools.prepare():
        warning(message)
    return tools


def load_persona(config: Config) -> tuple[dict[str, Persona], Persona | None]:
    personas = load_personas(config.personas_dir)
    if not config.persona:
        return personas, None
    if config.persona not in personas:
        raise PersonaError(unknown_persona(personas, config.persona) + f" (in {config.personas_dir})")
    return personas, personas[config.persona]


def run_chat(config: Config) -> int:
    try:
        personas, persona = load_persona(config)
    except PersonaError as e:
        error(str(e))
        return 1
    agent = Agent(config, persona=persona)
    try:
        agent.client.ensure_model(config.model)
        agent.tools = load_tools(config, agent.client)
    except LLMError as e:
        error(str(e))
        return 1

    session = Session(config, agent, personas=personas)
    if config.features["rag"]:
        session.index = load_existing_index(config, agent.client)
        agent.retriever = session.index
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
    if agent.tools:
        print(f"Tools: {', '.join(agent.tools.names())}. /tools to see what they do.")
    if agent.persona:
        print(f"Persona: {agent.persona.name}. /personas to list them, /persona ID to switch.")
    elif session.personas:
        print("No persona: /personas to list them, /persona ID to pick one.")

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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.env_path)
    except ConfigError as e:
        error(str(e))
        return 1
    if args.persona:
        config.persona = args.persona

    if args.command == "ingest":
        return run_ingest(config, args.paths, args.rebuild)
    if args.command == "status":
        return run_status(config)
    return run_chat(config)


if __name__ == "__main__":
    sys.exit(main())
