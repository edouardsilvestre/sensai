from dataclasses import dataclass, field

from sensai.core.agent import Agent
from sensai.core.approval import Approval
from sensai.core.cache import CachedAnswer, SemanticCache
from sensai.core.config import Config
from sensai.core.examples import Example, ExampleLibrary
from sensai.core.export.shares import ShareStore
from sensai.core.llm.ollama import ToolCall
from sensai.core.memory.conversation import Turn
from sensai.core.memory.profile import Profile
from sensai.core.memory.sessions import SessionError, SessionStore
from sensai.core.monitoring.log import InteractionLog
from sensai.core.persona import Persona
from sensai.core.prompts import PromptStore
from sensai.core.rag.index import DocumentIndex
from sensai.core.rag.models import SearchResult
from sensai.core.schedule import Schedule
from sensai.core.tools.questions import Asker
from sensai.interfaces.cli.output import warning

ASK_USER = "ask_user"
ALLOW_ANSWERS = {"y", "yes", "o", "oui"}
ALWAYS_ANSWERS = {"a", "always"}
REFUSE_ANSWERS = {"n", "no", "non"}


def ask_question(question: str, options: list[str]) -> str | None:
    print(f"[{ASK_USER}] {question}")
    for number, option in enumerate(options, start=1):
        print(f"  {number}. {option}")
    label = "Your answer (a number or your own words): " if options else "Your answer: "
    while True:
        try:
            answer = input(label).strip()
        except EOFError:
            print()
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer:
            return answer


@dataclass
class Session:
    config: Config
    agent: Agent
    index: DocumentIndex | None = None
    last_sources: list[SearchResult] = field(default_factory=list)
    # Found for the last message, but under SENSAI_RERANK_THRESHOLD: never sent to the model.
    last_dropped: list[SearchResult] = field(default_factory=list)
    store: SessionStore | None = None
    shares: ShareStore | None = None
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
    log: InteractionLog | None = None
    # Who answers the questions of the model: the page in the web UI, nobody in a run without a user.
    asker: Asker | None = ask_question


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


def confirm(question: str) -> bool:
    while True:
        try:
            answer = input(question).strip().lower()
        except EOFError:
            print()
            return False
        if answer in ALLOW_ANSWERS:
            return True
        if answer in REFUSE_ANSWERS:
            return False


def saved_fields(session: Session) -> dict:
    persona = session.agent.persona
    return {"persona": persona.id if persona else None, "prompt_version": session.agent.prompt_version}


def save_session(session: Session) -> None:
    memory = session.agent.memory
    if session.store is None or not session.config.features["sessions"] or not (memory.turns or memory.summary):
        return
    if session.session_id is None:
        session.session_id = session.agent.session_id or session.store.new_id()
    try:
        session.store.save(session.session_id, memory, session.config.model, **saved_fields(session))
    except SessionError as e:
        warning(f"The conversation is not saved: {e}")
