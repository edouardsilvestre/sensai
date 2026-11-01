import argparse
import json
import re
import shlex
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Callable, Generator
from pathlib import Path

from sensai.core.agent import INTERRUPTED, STOPPED, Agent, EmptyInputError
from sensai.core.approval import Approval
from sensai.core.cache import CachedAnswer, CacheError, SemanticCache
from sensai.core.config import API_KEY_HELP, APPROVAL_MODES, FEATURES, Config, ConfigError, load_config
from sensai.core.events import (
    AgentStartedEvent,
    CacheFailedEvent,
    CacheHitEvent,
    CompactionEvent,
    CompactionFailedEvent,
    CompactionStartedEvent,
    DocumentEvent,
    DocumentFailedEvent,
    DoneEvent,
    Event,
    ExamplesEvent,
    ExamplesFailedEvent,
    GuardrailEvent,
    ReasoningStoppedEvent,
    ReviewDoneEvent,
    SourcesEvent,
    ThoughtEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.document.document import Edit, EditReport
from sensai.core.examples import Example, ExampleError, ExampleLibrary, make_example
from sensai.core.guardrails.rules import STAGES, GuardrailError, Guardrails, Hit, load_rules, model_judge
from sensai.core.llm.ollama import LLMError, OllamaClient, ToolCall
from sensai.core.memory.conversation import Turn
from sensai.core.memory.profile import Profile, ProfileError
from sensai.core.memory.sessions import SessionError, SessionStore
from sensai.core.memory.store import MemoryStoreError
from sensai.core.orchestrator import Review, ReviewError
from sensai.core.persona import Persona, PersonaError, load_personas
from sensai.core.prompts import PromptError, PromptStore, PromptVersion, VersionScore, compare, comparison, diff
from sensai.core.rag.embedder import Embedder
from sensai.core.rag.index import DocumentIndex, IngestProgress, IngestReport
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers
from sensai.core.rag.store import StoreError, VectorStore
from sensai.core.schedule import WHEN_FORMS, Schedule, ScheduleError, run_task
from sensai.core.structured import ANSWER_SCHEMA, StructuredError, cited_sources, parse_answer
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.mcp import MCPConfigError
from sensai.core.tools.permissions import ALWAYS_DENIED, Permissions
from sensai.core.tools.registry import ToolRegistry

EXIT_COMMANDS = {"/exit", "/quit"}
POLL_SECONDS = 15
SCHEDULE_ADD = re.compile(r"(every \S+|daily \S+|weekly \S+ \S+)(?: --allow (\S+))? (.+)", re.IGNORECASE)
SCHEDULE_USAGE = f"Usage: /schedule add WHEN [--allow TOOL,TOOL] TASK, or /schedule remove N. WHEN is {WHEN_FORMS}"
ALLOW_ANSWERS = {"y", "yes", "o", "oui"}
ALWAYS_ANSWERS = {"a", "always"}
REFUSE_ANSWERS = {"n", "no", "non"}
APPROVAL_DESCRIPTIONS = {
    "critical": "asks before the tools that send data out of the machine or run code",
    "all": "asks before every tool call",
    "off": "never asks",
}
SHORT_TEXT = 70
HELP = """\
Commands:
  /ingest PATH...         index files or folders so that answers can draw on them
  /sources                show the excerpts retrieved for the last message
  /tools                  list the tools the model can call
  /files                  show the folders the model may read and write
  /context                show the context usage and the summary of older turns
  /sessions               list the saved conversations
  /resume [ID]            continue a saved conversation (the last one without ID)
  /new                    start a new conversation, the current one stays saved
  /history                list the messages of this branch with their numbers
  /edit N [TEXT]          replace message N in a new branch; the original conversation stays in its branch
  /branches               list the branches of this conversation
  /branch N               continue branch N
  /profile                show the instructions the model gets about you
  /profile add TEXT       add an instruction to your profile
  /profile remove N       remove instruction N from your profile
  /document               show the working document the model keeps up to date
  /document undo          undo the changes made to the document by the last answer
  /document export PATH   write the document to a Markdown file
  /features               list the features and whether they are on
  /feature NAME on|off    turn a feature on or off for this session
  /personas               list the personas
  /persona [ID]           show the active persona, or switch to another one (the history is kept)
  /review QUESTION        the investigator answers, the fact-checker checks, the editor writes the text
  /json QUESTION          answer in JSON: the answer, its claims with their excerpts, the entities and the events
  /guardrails             list the rules checking your messages, the answers and the tool calls
  /approval [MODE]        show or change when to ask before running a tool: critical, all or off
  /reasoning [N]          list the questions reasoned about, or show the steps of question N
  /cache [clear]          show the answers kept to reuse, or remove them all
  /examples               list the answers kept as examples, * marks those shown for the last answer
  /example add|remove N   keep the last answer as an example for similar questions, or remove example N
  /prompts                list the versions of the system prompt, * marks the one in use
  /prompt show|use N      read version N of the system prompt, or use it from now on (roll back)
  /prompt diff N M        show what changed between versions N and M
  /prompt compare N M     ask the test cases with versions N and M, and compare the answers, time and tokens
  /schedule               list the tasks run on their own by python -m sensai schedule
  /schedule add WHEN TASK add a task: every 6h, daily 08:00 or weekly mon 08:00, --allow TOOL for a critical tool
  /schedule remove N      remove task N, its reports stay
  /reports [N]            list the reports of the scheduled tasks, or read report N
  /exit, /quit            leave (or press Ctrl+D)
Ctrl+C during an answer stops it: what was written is kept, and your next message redirects it."""


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
    schedule_parser = commands.add_parser("schedule", help="run the scheduled tasks when they are due")
    schedule_parser.add_argument("--once", action="store_true", help="run what is due and exit, for cron or the like")
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
    cache: SemanticCache | None = None
    examples: ExampleLibrary | None = None
    examples_used: list[Example] = field(default_factory=list)
    answered: tuple[Turn, list[SearchResult]] | None = None
    prompts: PromptStore | None = None
    schedule: Schedule | None = None


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


def confirm_cached(cached: CachedAnswer) -> bool:
    when = cached.created_at.astimezone().strftime("%Y-%m-%d at %H:%M")
    print(f'[cache] A close question was answered on {when}: "{cached.question}" (similarity {cached.similarity:.0%})')
    while True:
        try:
            answer = input("Use this answer? [y]es / [n]o: ").strip().lower()
        except EOFError:
            print()
            return False
        if answer in ALLOW_ANSWERS:
            return True
        if answer in REFUSE_ANSWERS:
            return False


def prompt(session: Session) -> str:
    persona = session.agent.persona
    return f"\n[{persona.id}] > " if persona else "\n> "


def print_sources(sources: list[SearchResult], cited: list[int]) -> None:
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
        session.examples_used = []
    turns = session.agent.memory.turns
    last = turns[-1] if turns else None
    show_events(session, session.agent.run(line))
    turns = session.agent.memory.turns
    if turns and turns[-1] is not last:
        session.answered = (turns[-1], list(session.last_sources))


def show_events(
    session: Session,
    events: Generator[Event, None, None],
    structured: bool = False,
    interrupt: Callable[[], bool] | None = None,
) -> None:
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
            elif isinstance(event, CacheHitEvent):
                print("[cache] Answer reused, the model was not called.")
            elif isinstance(event, CacheFailedEvent):
                warning(f"The cache could not be used: {event.message}")
            elif isinstance(event, ExamplesEvent):
                session.examples_used = event.examples
                questions = ", ".join(f'"{example.question}"' for example in event.examples)
                print(f"[examples] {plural(len(event.examples), 'example')}: {questions}")
            elif isinstance(event, ExamplesFailedEvent):
                warning(f"No examples were shown to the model: {event.message}")
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
                # A JSON answer is printed once complete and checked, indented.
                if structured:
                    continue
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
                if structured:
                    print_structured(event.content, sources)
                    continue
                print()
                if sources:
                    print_sources(sources, cited_numbers(event.content))
            elif isinstance(event, DocumentEvent):
                print_document_report(event.report)
            elif isinstance(event, DocumentFailedEvent):
                warning(f"The document was not updated: {event.message}")
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
        # Closes the connection too: Ollama stops generating.
        events.close()
        if session.config.features["branching"] and (interrupt or session.agent.interrupt)():
            print("\n[interrupted] What was written is kept: tell the model how to go on, or /edit to change the question.")
        else:
            print("\n[interrupted]")
    save_session(session)


CHANGES = {"append": "added to {}", "replace": "corrected in {}", "rewrite": "{} rewritten", "add_section": "new section {}"}


def describe_edits(edits: list[Edit]) -> str:
    # One entry per section and kind of change: "2 added to Timeline" rather than the same line twice.
    counts: dict[str, int] = {}
    for edit in edits:
        change = CHANGES[edit.op].format(" ".join(edit.section.split()))
        counts[change] = counts.get(change, 0) + 1
    return ", ".join(f"{count} {change}" if count > 1 else change for change, count in counts.items())


def print_document_report(report: EditReport) -> None:
    if report.applied:
        print(f"[document] {describe_edits(report.applied)}. /document to read it.")
    for rejected in report.rejected:
        warning(f"document edit rejected ({rejected.edit.describe()}): {rejected.reason}")


def save_session(session: Session) -> None:
    memory = session.agent.memory
    if session.store is None or not session.config.features["sessions"] or not (memory.turns or memory.summary):
        return
    if session.session_id is None:
        session.session_id = session.store.new_id()
    try:
        persona = session.agent.persona
        session.store.save(
            session.session_id, memory, session.config.model,
            persona.id if persona else None, session.agent.prompt_version,
        )
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
    memory_turns = session.agent.memory.turns
    session.answered = (memory_turns[-1], []) if memory_turns else None
    summary = ", plus a summary of older turns" if session.agent.memory.summary else ""
    print(f"Resumed conversation {session_id}: {plural(turns, 'turn')}{summary}. /context shows them.")
    branches = session.agent.memory.branches()
    if len(branches) > 1:
        print(f"On branch {session.agent.memory.branch.number} of {len(branches)}: /branches lists them.")
    return True


def resume_command(session: Session, argument: str) -> None:
    if sessions_enabled(session):
        # The current conversation was saved after its last message: switching loses nothing.
        resume(session, argument or "last")


def new_command(session: Session) -> None:
    previous = session.session_id
    session.agent.memory.clear()
    session.session_id = None
    session.last_sources = []
    session.agent.reasoning.clear()
    session.answered = None
    saved = f" Conversation {previous} is saved: /resume {previous} to come back to it." if previous else ""
    print(f"New conversation.{saved}")


def shorten(text: str) -> str:
    return textwrap.shorten(text, width=SHORT_TEXT, placeholder="...")


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
        save_session(session)
        last = f" Last answer: {shorten(memory.turns[-1].assistant)}" if memory.turns else ""
        print(f"On branch {argument}, {plural(len(memory.turns), 'turn')}.{last}")


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
            if not value.strip().isdecimal():
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
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(memory.document.markdown(), encoding="utf-8")
        except OSError as e:
            error(f"Cannot write {path}: {e}")
            return
        print(f"Document written to {path}.")
    elif action:
        error("Usage: /document [undo | export PATH]")
    else:
        show_document(session)


def show_document(session: Session) -> None:
    document = session.agent.memory.document
    if not session.config.features["document"]:
        print("The document feature is off: /feature document on to update it after every answer.")
    if document.is_empty:
        print(f"The document is empty. It fills in as the conversation establishes facts: {', '.join(document.names())}.")
        return
    print(document.markdown().rstrip())


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


def folders(permissions: Permissions, paths: list[Path]) -> str:
    shown = [permissions.display(path) + ("" if path.is_dir() else " (missing)") for path in paths]
    return ", ".join(shown) or "none"


def files_command(session: Session) -> None:
    config = session.config
    if not config.features["files"]:
        print("The files feature is off: /feature files on to let the model read and write files.")
    permissions = Permissions(config.files_read, config.files_write, config.files_deny)
    print(f"Read (SENSAI_FILES_READ): {folders(permissions, permissions.read)}")
    print(f"Read and write (SENSAI_FILES_WRITE): {folders(permissions, permissions.write)}")
    print(f"Always denied: {', '.join(ALWAYS_DENIED + tuple(permissions.deny))}")
    if config.approval != "off":
        print("Sensai asks before every file written: /approval to change it.")


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
    if name == "cache" and value == "on" and session.cache is None:
        session.cache = session.agent.cache = load_cache(config, session.agent.client)
    if name == "examples" and value == "on" and session.examples is None:
        session.examples = session.agent.examples = load_examples(config, session.agent.client)
    if name == "prompt_versions":
        if value == "on" and session.prompts is None:
            session.prompts = session.agent.prompts = load_prompts(config)
        if value == "on" and session.prompts is not None and session.agent.persona is None:
            active = session.prompts.active()
            if active:
                session.agent.prompt_version = active.number
                session.agent.memory.system_prompt = active.prompt
        elif value == "off" and session.agent.persona is None:
            session.agent.prompt_version = None
            session.agent.memory.system_prompt = config.system_prompt
    if name == "schedule" and value == "on" and session.schedule is None:
        session.schedule = load_schedule(config)


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
    show_events(session, review.run(argument), interrupt=review.interrupt)


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
    if not argument.isdecimal() or not 1 <= int(argument) <= len(history):
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


def cache_command(session: Session, argument: str) -> None:
    if argument not in ("", "clear"):
        error("Usage: /cache [clear]")
        return
    config, cache = session.config, session.cache
    if cache is None:
        if config.features["cache"]:
            print("The cache could not be opened: /feature cache on to try again.")
        else:
            print("The cache feature is off: /feature cache on to reuse the answers to close questions.")
        return
    if argument == "clear":
        print(f"Cache cleared: {plural(cache.clear(), 'answer')} removed.")
        return
    stats = cache.stats()
    print(
        f"Cache: {plural(stats.answers, 'answer')} kept, reused {plural(stats.uses, 'time')} "
        f"({config.cache_path}, threshold {config.cache_threshold}%)."
    )
    if not config.features["cache"]:
        print("The cache feature is off: /feature cache on to reuse them.")


def kept_on(example: Example) -> str:
    if not example.created_at:
        return ""
    return f" (kept on {datetime.fromisoformat(example.created_at).astimezone():%Y-%m-%d})"


def examples_off(session: Session) -> None:
    if not session.config.features["examples"]:
        print("The examples feature is off: /feature examples on to show them to the model.")


def list_examples(session: Session) -> None:
    if session.examples is None:
        print("The examples could not be loaded: /feature examples on to try again.")
        return
    examples = session.examples.examples
    if not examples:
        print("No examples yet: /example add after a good answer to keep it.")
    else:
        print("Answers kept as examples (* shown to the model for the last answer):")
        for number, example in enumerate(examples, start=1):
            mark = "*" if any(example is used for used in session.examples_used) else " "
            print(f"  {mark} {number}. {example.question}{kept_on(example)}")
    examples_off(session)


def add_example(session: Session) -> None:
    turns = session.agent.memory.turns
    if session.answered is None or not turns or turns[-1] is not session.answered[0]:
        error("No answer to keep yet: ask a question, then /example add to keep its answer.")
        return
    turn, sources = session.answered
    if STOPPED in turn.assistant:
        error("An answer stopped by a guardrail cannot be kept as an example.")
        return
    if INTERRUPTED in turn.assistant:
        error("An interrupted answer cannot be kept as an example.")
        return
    try:
        example = make_example(turn.user, turn.assistant, sources)
        session.examples.add(example)
    except ExampleError as e:
        error(str(e))
        return
    print(f'Answer kept as example {len(session.examples.examples)}: "{example.question}"')
    examples_off(session)


def example_command(session: Session, argument: str) -> None:
    action, _, number = argument.partition(" ")
    number = number.strip()
    if not (action == "add" and not number) and not (action == "remove" and number.isdecimal()):
        error("Usage: /example add, or /example remove N")
        return
    if session.examples is None:
        print("The examples could not be loaded: /feature examples on to try again.")
        return
    if action == "add":
        add_example(session)
        return
    try:
        removed = session.examples.remove(int(number))
    except ExampleError as e:
        error(str(e))
        return
    print(f'Example {number} removed: "{removed.question}"')


def created_on(version: PromptVersion) -> str:
    if not version.created_at:
        return "-"
    return f"{datetime.fromisoformat(version.created_at).astimezone():%Y-%m-%d %H:%M}"


def list_prompts(session: Session) -> None:
    if not session.config.features["prompt_versions"]:
        print("The prompt_versions feature is off: /feature prompt_versions on to keep the system prompt.")
        return
    if session.prompts is None:
        print("The prompt versions could not be loaded: /feature prompt_versions on to try again.")
        return
    try:
        history = session.prompts.history()
    except PromptError as e:
        error(str(e))
        return
    if not history.versions:
        print("No version of the system prompt yet: it is kept at the next start.")
        return
    print("Versions of the system prompt (* in use):")
    for version in reversed(history.versions):
        mark = "*" if version.number == history.active else " "
        size = plural(len(version.prompt), "char")
        print(f"  {mark} {version.number}. {created_on(version)}  {version.hash}  {size}")
    if history.comparisons:
        last = history.comparisons[-1]
        scores = ", ".join(f"version {s['version']} {s['passed']}/{s['cases']}" for s in last["scores"])
        print(f"Last comparison ({last['model']}): {scores}.")
    print("/prompt show N to read one, /prompt use N to roll back to it.")


def print_scores(scores: list[VersionScore]) -> None:
    print(f"\n{'Version':<9}{'Passed':<9}{'Avg time':<10}Avg tokens")
    for version in scores:
        passed = f"{version.passed}/{len(version.results)}"
        print(f"{version.version:<9}{passed:<9}{version.seconds:<10.2f}{version.tokens:.0f}")
    for version in scores:
        for number, result in enumerate(version.results, start=1):
            if result.failed:
                failed = ", ".join(result.failed)
                print(f"  version {version.version}, case {number} ({result.case.question}): {failed}")


def compare_prompts(session: Session, versions: list[PromptVersion]) -> None:
    config = session.config
    numbers = " and ".join(str(version.number) for version in versions)
    print(f"Comparing versions {numbers} on the test cases with {config.model}...")
    try:
        scores = compare(session.agent.client, config, versions)
    except (PromptError, LLMError) as e:
        error(str(e))
        return
    except KeyboardInterrupt:
        print("\n[interrupted] The comparison is not kept.")
        return
    print_scores(scores)
    try:
        session.prompts.record(comparison(config.model, scores))
    except PromptError as e:
        warning(f"The comparison is not kept: {e}")


PROMPT_USAGE = "Usage: /prompt show|use N, or /prompt diff|compare N M"


def prompt_command(session: Session, argument: str) -> None:
    words = argument.split()
    action, numbers = (words[0], words[1:]) if words else ("", [])
    count = {"show": 1, "use": 1, "diff": 2, "compare": 2}.get(action)
    if count is None or len(numbers) != count or not all(n.isdecimal() for n in numbers):
        error(PROMPT_USAGE)
        return
    if not session.config.features["prompt_versions"]:
        print("The prompt_versions feature is off: /feature prompt_versions on to keep the system prompt.")
        return
    if session.prompts is None:
        print("The prompt versions could not be loaded: /feature prompt_versions on to try again.")
        return
    try:
        versions = [session.prompts.version(int(number)) for number in numbers]
    except PromptError as e:
        error(str(e))
        return
    if action == "show":
        print(f"Version {versions[0].number} of the system prompt ({created_on(versions[0])}):")
        print(textwrap.indent(versions[0].prompt, "  "))
    elif action == "diff":
        print(diff(*versions) or f"Versions {versions[0].number} and {versions[1].number} are identical.")
    elif action == "compare":
        compare_prompts(session, versions)
    else:
        try:
            session.prompts.activate(versions[0].number)
        except PromptError as e:
            error(str(e))
            return
        agent = session.agent
        agent.prompt_version = versions[0].number
        agent.memory.system_prompt = versions[0].prompt
        print(f"Sensai uses version {versions[0].number} of its system prompt from now on.")


def print_structured(answer: str, sources: list[SearchResult]) -> None:
    try:
        data = parse_answer(answer)
    except StructuredError as e:
        error(f"{e}:\n{answer}")
        return
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if sources:
        print_sources(sources, cited_sources(data))


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


def json_command(session: Session, argument: str) -> None:
    if not argument:
        error("Usage: /json QUESTION")
        return
    session.last_sources = []
    show_events(session, session.agent.run(argument, ANSWER_SCHEMA), structured=True)


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
    elif name == "/files":
        files_command(session)
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
    elif name == "/history":
        history_command(session)
    elif name == "/edit":
        edit_command(session, argument)
    elif name == "/branches":
        branches_command(session)
    elif name == "/branch":
        branch_command(session, argument)
    elif name == "/profile":
        profile_command(session, argument)
    elif name == "/document":
        document_command(session, argument)
    elif name == "/guardrails":
        guardrails_command(session)
    elif name == "/approval":
        approval_command(session, argument.lower())
    elif name == "/reasoning":
        reasoning_command(session, argument)
    elif name == "/cache":
        cache_command(session, argument.lower())
    elif name == "/examples":
        list_examples(session)
    elif name == "/example":
        example_command(session, argument.lower())
    elif name == "/prompts":
        list_prompts(session)
    elif name == "/prompt":
        prompt_command(session, argument)
    elif name == "/schedule":
        schedule_command(session, argument)
    elif name == "/reports":
        reports_command(session, argument)
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
    elif name == "/json":
        json_command(session, argument)
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


def load_cache(config: Config, client: OllamaClient) -> SemanticCache | None:
    try:
        client.ensure_model(config.embed_model)
        embedder = Embedder(client, config.embed_model, max_chars=config.chunk_size)
        return SemanticCache(config.cache_path, embedder.embed_query, config.cache_threshold / 100)
    except (LLMError, CacheError) as e:
        warning(f"Cache disabled: {e}")
        return None


def load_examples(config: Config, client: OllamaClient) -> ExampleLibrary | None:
    embedder = Embedder(client, config.embed_model, max_chars=config.chunk_size)
    try:
        examples = ExampleLibrary(config.examples_path, embedder.embed_query)
        if examples.examples and config.features["examples"]:
            client.ensure_model(config.embed_model)
    except (ExampleError, LLMError) as e:
        warning(f"Examples disabled: {e}")
        return None
    return examples


def load_prompts(config: Config) -> PromptStore | None:
    prompts = PromptStore(config.prompts_dir)
    try:
        if config.system_prompt:
            version = prompts.snapshot(config.system_prompt)
            if version is not None and version.number > 1:
                print(f"System prompt changed: kept as version {version.number}. /prompts to list them.")
    except PromptError as e:
        warning(f"Prompt versions disabled: {e}")
        return None
    return prompts


def load_schedule(config: Config) -> Schedule | None:
    try:
        return Schedule(config.schedule_path, config.reports_dir)
    except ScheduleError as e:
        warning(f"Schedule disabled: {e}")
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


def start_session(config: Config) -> Session | None:
    """The model checked, then the tools, the rules, the documents, the examples. None when Sensai cannot start."""
    try:
        personas, persona = load_persona(config)
    except PersonaError as e:
        error(str(e))
        return None
    prompts = load_prompts(config) if config.features["prompt_versions"] else None
    agent = Agent(config, persona=persona, prompts=prompts)
    try:
        agent.client.ensure_model(config.model)
        agent.tools = load_tools(config, agent.client)
        agent.guardrails = load_guardrails(config, agent.client)
    except (LLMError, GuardrailError, MemoryStoreError, MCPConfigError) as e:
        error(str(e))
        return None

    session = Session(config, agent, personas=personas, prompts=prompts)
    if config.features["rag"]:
        session.index = load_existing_index(config, agent.client)
        agent.retriever = session.index
    session.examples = agent.examples = load_examples(config, agent.client)
    return session


def run_chat(config: Config, resume_id: str | None = None) -> int:
    session = start_session(config)
    if session is None:
        return 1
    agent = session.agent
    if config.features["cache"]:
        session.cache = agent.cache = load_cache(config, agent.client)
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
        version = f" (prompt version {agent.prompt_version})" if agent.prompt_version else ""
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
    session = start_session(config)
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
    if args.command == "schedule":
        return run_schedule(config, args.once)
    return run_chat(config, args.resume)


if __name__ == "__main__":
    sys.exit(main())
