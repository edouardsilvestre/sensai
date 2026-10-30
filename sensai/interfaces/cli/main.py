import argparse
import json
import shlex
import sys
import textwrap
from dataclasses import dataclass, field
from collections.abc import Iterator
from pathlib import Path

from sensai.core.agent import Agent, EmptyInputError
from sensai.core.approval import Approval
from sensai.core.config import API_KEY_HELP, APPROVAL_MODES, FEATURES, Config, ConfigError, load_config
from sensai.core.events import (
    AgentStartedEvent,
    CompactionEvent,
    CompactionFailedEvent,
    CompactionStartedEvent,
    DoneEvent,
    Event,
    GuardrailEvent,
    ReasoningStoppedEvent,
    ReviewDoneEvent,
    SourcesEvent,
    ThoughtEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.guardrails.rules import STAGES, GuardrailError, Guardrails, Hit, load_rules, model_judge
from sensai.core.llm.ollama import LLMError, OllamaClient, ToolCall
from sensai.core.memory.profile import Profile, ProfileError
from sensai.core.memory.sessions import SessionError, SessionStore
from sensai.core.memory.store import MemoryStoreError
from sensai.core.orchestrator import Review, ReviewError
from sensai.core.persona import Persona, PersonaError, load_personas
from sensai.core.rag.index import DocumentIndex, IngestProgress, IngestReport
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers
from sensai.core.rag.store import StoreError, VectorStore
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.mcp import MCPConfigError
from sensai.core.tools.registry import ToolRegistry

EXIT_COMMANDS = {"/exit", "/quit"}
ALLOW_ANSWERS = {"y", "yes", "o", "oui"}
ALWAYS_ANSWERS = {"a", "always"}
REFUSE_ANSWERS = {"n", "no", "non"}
APPROVAL_DESCRIPTIONS = {
    "critical": "asks before the tools that send data out of the machine or run code",
    "all": "asks before every tool call",
    "off": "never asks",
}
HELP = """\
Commands:
  /ingest PATH...         index files or folders so that answers can draw on them
  /sources                show the excerpts retrieved for the last message
  /tools                  list the tools the model can call
  /context                show the context usage and the summary of older turns
  /sessions               list the saved conversations
  /resume [ID]            continue a saved conversation (the last one without ID)
  /new                    start a new conversation, the current one stays saved
  /profile                show the instructions the model gets about you
  /profile add TEXT       add an instruction to your profile
  /profile remove N       remove instruction N from your profile
  /features               list the features and whether they are on
  /feature NAME on|off    turn a feature on or off for this session
  /personas               list the personas
  /persona [ID]           show the active persona, or switch to another one (the history is kept)
  /review QUESTION        the investigator answers, the fact-checker checks, the editor writes the text
  /guardrails             list the rules checking your messages, the answers and the tool calls
  /approval [MODE]        show or change when to ask before running a tool: critical, all or off
  /reasoning [N]          list the questions reasoned about, or show the steps of question N
  /exit, /quit            leave (or press Ctrl+D)"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sensai", description="Sensai CLI chatbot")
    parser.add_argument("-e", "--env-file", dest="env_path", type=Path, help="Path to a .env file (default: ./.env)")
    parser.add_argument(
        "-r", "--resume", nargs="?", const="last", metavar="ID",
        help="continue a saved conversation (the last one without ID)",
    )
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
    store: SessionStore | None = None
    # Given at the first save, so that a conversation without any message is never saved.
    session_id: str | None = None
    profile: Profile | None = None
    always_allowed: set[str] = field(default_factory=set)
    personas: dict[str, Persona] = field(default_factory=dict)


def ask_approval(session: Session, call: ToolCall, reason: str) -> Approval:
    if call.name in session.always_allowed:
        print(f"[{call.name}] allowed for this session")
        return Approval(True)
    question = f"Allow {call.name}? {reason[:1].upper()}{reason[1:]}. [y]es / [n]o / [a]lways this session: "
    while True:
        try:
            answer = input(question).strip().lower()
        except EOFError:
            print()
            return Approval(False, "no answer")
        if answer in ALLOW_ANSWERS:
            return Approval(True)
        if answer in ALWAYS_ANSWERS:
            session.always_allowed.add(call.name)
            return Approval(True)
        if answer in REFUSE_ANSWERS:
            try:
                why = input("Why? (optional, sent to the model): ").strip()
            except EOFError:
                print()
                why = ""
            return Approval(False, why)


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


WHERE = {"input": "your message", "output": "the answer", "tools": "a tool call"}


def print_guardrail(hit: Hit) -> None:
    where, found = WHERE[hit.stage], f" ({hit.found})" if hit.found else ""
    if hit.action == "redact":
        print(f"[guardrail] {hit.rule}: {plural(hit.count, 'passage')} redacted in {where}{found}")
    elif hit.action == "warn":
        warning(f"guardrail {hit.rule}: found in {where}{found}" + (f". {hit.message}" if hit.message else ""))
    elif hit.stage == "input":
        warning(f"Message blocked by the guardrail {hit.rule}: {hit.message or hit.found}. Nothing was sent to the model.")
    elif hit.stage == "output":
        warning(f"Answer stopped by the guardrail {hit.rule}{found}" + (f": {hit.message}" if hit.message else ""))
    # A blocked tool call is reported by its result, which the model also gets.


def chat_turn(session: Session, line: str) -> None:
    if line.strip():
        session.last_sources = []
    show_events(session, session.agent.run(line))


def show_events(session: Session, events: Iterator[Event]) -> None:
    sources: list[SearchResult] = []
    mid_line = False
    try:
        for event in events:
            if isinstance(event, AgentStartedEvent):
                if mid_line:
                    print()
                    mid_line = False
                sources = []
                print(f"\n── {event.persona.id} ({event.persona.name}) ──")
            elif isinstance(event, ReviewDoneEvent):
                rounds = plural(event.rounds, "check")
                if event.approved:
                    print(f"\n[review] approved by the fact-checker after {rounds}, the editor's text is kept.")
                else:
                    warning(f"Answer not approved after {rounds}: the last answer is kept, check the remarks above.")
            elif isinstance(event, SourcesEvent):
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
            elif isinstance(event, GuardrailEvent):
                if mid_line:
                    print()
                    mid_line = False
                print_guardrail(event.hit)
            elif isinstance(event, ThoughtEvent):
                if mid_line:
                    print()
                    mid_line = False
                print(f"[thought {event.number}] {event.thought}")
            elif isinstance(event, ReasoningStoppedEvent):
                warning(f"Reasoning stopped: {event.reason}. Sensai answers from what it found.")
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
    session.agent.reasoning.clear()
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
    session.agent.reasoning.clear()
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
    if ingest(session.index, paths) and not session.config.features["rag"]:
        print("The rag feature is off: /feature rag on to answer from these documents.")


def tools_command(session: Session) -> None:
    agent = session.agent
    if not session.config.features["tools"]:
        print("The tools feature is off: /feature tools on to let the model call tools.")
        return
    if not agent.tools:
        print("No tool is enabled, turn one on with /feature NAME on:")
        for name, feature in FEATURES.items():
            if feature.tool:
                print(f"  {name}: {feature.description}")
        return
    print("Tools:")
    for name, description in agent.tools.descriptions().items():
        print(f"  {name}: {description}")


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


def guardrails_source(config: Config) -> str:
    return str(config.guardrails_path) if config.guardrails_path.exists() else "default rules"


def guardrails_command(session: Session) -> None:
    config = session.config
    if not config.features["guardrails"]:
        print("The guardrails feature is off: /feature guardrails on to check messages, answers and tool calls.")
        return
    rules = session.agent.guardrails.rules
    print(f"Guardrails ({guardrails_source(config)}):")
    if not rules:
        print("  no rule")
    for rule in rules:
        tools = f" ({', '.join(sorted(rule.tools))})" if rule.tools else ""
        where = ", ".join(stage + (tools if stage == "tools" else "") for stage in STAGES if stage in rule.stages)
        print(f"  {rule.name}: {rule.action} in {where}, {rule.summary}")
    if not config.guardrails_path.exists():
        print(f"Write your own rules in {config.guardrails_path}: see guardrails.example.json.")
def approval_command(session: Session, argument: str) -> None:
    if argument:
        if argument not in APPROVAL_MODES:
            error(f"Usage: /approval [{'|'.join(APPROVAL_MODES)}]")
            return
        session.config.approval = argument
        session.always_allowed.clear()
    mode = session.config.approval
    print(f"Approval: {mode}, Sensai {APPROVAL_DESCRIPTIONS[mode]}.")
    if session.always_allowed:
        print(f"Allowed for this session: {', '.join(sorted(session.always_allowed))}.")


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
    if (name == "tools" or FEATURES[name].tool) and config.features["tools"]:
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
    except (LLMError, MemoryStoreError, MCPConfigError) as e:
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


def review_command(session: Session, argument: str) -> None:
    if not session.config.features["orchestration"]:
        print("The orchestration feature is off: /feature orchestration on to review answers.")
        return
    if not argument:
        error("Usage: /review QUESTION")
        return
    try:
        review = Review(session.agent, session.personas)
    except ReviewError as e:
        error(f"{e} (in {session.config.personas_dir}).")
        return
    session.last_sources = []
    show_events(session, review.run(argument))


def reasoning_command(session: Session, argument: str) -> None:
    history = session.agent.reasoning
    if not history:
        hint = "ask a question" if session.config.features["reasoning"] else "/feature reasoning on, then ask one"
        print(f"No reasoning yet in this conversation: {hint}.")
        return
    if not argument:
        print("Reasoning of this conversation:")
        for number, reasoning in enumerate(history, start=1):
            print(f"  {number}. {reasoning.question} ({plural(len(reasoning.steps), 'step')})")
        print("/reasoning N to read the steps of question N.")
        return
    if not argument.isdigit() or not 1 <= int(argument) <= len(history):
        error(f"Usage: /reasoning [N], with N between 1 and {len(history)}")
        return
    reasoning = history[int(argument) - 1]
    print(f"Question {argument}: {reasoning.question}")
    for number, step in enumerate(reasoning.steps, start=1):
        print(f"[thought {number}] {step.thought}")
        if not step.final:
            print(textwrap.indent(describe_call(step.call()), "  "))
        if step.result is not None:
            print(textwrap.indent(step.result.content, "    "))


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
    elif name == "/sessions":
        sessions_command(session)
    elif name == "/resume":
        resume_command(session, argument)
    elif name == "/new":
        new_command(session)
    elif name == "/profile":
        profile_command(session, argument)
    elif name == "/guardrails":
        guardrails_command(session)
    elif name == "/approval":
        approval_command(session, argument.lower())
    elif name == "/reasoning":
        reasoning_command(session, argument)
    elif name == "/features":
        list_features(session)
    elif name == "/feature":
        feature_command(session, argument)
    elif name == "/personas":
        list_personas(session)
    elif name == "/persona":
        persona_command(session, argument)
    elif name == "/review":
        review_command(session, argument)
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
    for message in tools.warnings + tools.prepare():
        warning(message)
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
def load_guardrails(config: Config, client: OllamaClient) -> Guardrails:
    """Raises GuardrailError for an invalid rules file: Sensai does not start with rules it did not understand."""
    rules = load_rules(config.guardrails_path)
    if config.features["guardrails"]:
        print(f"Guardrails: {plural(len(rules), 'rule')} from {guardrails_source(config)}. /guardrails to list them.")
    return Guardrails(rules, model_judge(client, config))


def load_persona(config: Config) -> tuple[dict[str, Persona], Persona | None]:
    personas = load_personas(config.personas_dir)
    if not config.persona:
        return personas, None
    if config.persona not in personas:
        raise PersonaError(unknown_persona(personas, config.persona) + f" (in {config.personas_dir})")
    return personas, personas[config.persona]


def run_chat(config: Config, resume_id: str | None = None) -> int:
    try:
        personas, persona = load_persona(config)
    except PersonaError as e:
        error(str(e))
        return 1
    agent = Agent(config, persona=persona)
    try:
        agent.client.ensure_model(config.model)
        agent.tools = load_tools(config, agent.client)
        agent.guardrails = load_guardrails(config, agent.client)
    except (LLMError, GuardrailError, MemoryStoreError, MCPConfigError) as e:
        error(str(e))
        return 1

    session = Session(config, agent, personas=personas)
    if config.features["rag"]:
        session.index = load_existing_index(config, agent.client)
        agent.retriever = session.index
    agent.approver = lambda call, reason: ask_approval(session, call, reason)
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
    load_memory(session, resume_id)

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
    return run_chat(config, args.resume)


if __name__ == "__main__":
    sys.exit(main())
