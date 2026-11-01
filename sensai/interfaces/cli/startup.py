from sensai.core.agent import Agent
from sensai.core.cache import CacheError, SemanticCache
from sensai.core.config import Config
from sensai.core.examples import ExampleError, ExampleLibrary
from sensai.core.export.shares import ShareStore
from sensai.core.guardrails.rules import GuardrailError, Guardrails, load_rules, model_judge
from sensai.core.llm.ollama import LLMError, OllamaClient
from sensai.core.memory.profile import Profile, ProfileError
from sensai.core.memory.sessions import SessionStore
from sensai.core.memory.store import MemoryStoreError
from sensai.core.monitoring.log import InteractionLog, LogError
from sensai.core.persona import Persona, PersonaError, load_personas
from sensai.core.prompts import PromptError, PromptStore
from sensai.core.rag.embedder import Embedder
from sensai.core.rag.index import DocumentIndex
from sensai.core.rag.store import StoreError
from sensai.core.schedule import Schedule, ScheduleError
from sensai.core.tools.catalog import build_registry
from sensai.core.tools.mcp import MCPConfigError
from sensai.core.tools.questions import Asker
from sensai.core.tools.registry import ToolRegistry
from sensai.interfaces.cli.commands.conversation import resume
from sensai.interfaces.cli.output import error, plural, warning
from sensai.interfaces.cli.session import Session, ask_question


def open_index(config: Config, client: OllamaClient, rebuild: bool = False) -> DocumentIndex:
    client.ensure_model(config.embed_model)
    return DocumentIndex.open(config, client, rebuild=rebuild)


def guardrails_source(config: Config) -> str:
    return str(config.guardrails_path) if config.guardrails_path.exists() else "default rules"


def reload_tools(session: Session) -> None:
    try:
        session.agent.tools = load_tools(session.config, session.agent.client, session.asker)
    except (LLMError, MemoryStoreError, MCPConfigError) as e:
        error(str(e))
        return
    if session.agent.tools:
        print(f"Tools: {', '.join(session.agent.tools.names())}. /tools to see what they do.")


def unknown_persona(personas: dict[str, Persona], persona_id: str) -> str:
    return f"Unknown persona '{persona_id}'. Available: {', '.join(personas) or 'none'}"


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


def load_log(config: Config) -> InteractionLog | None:
    try:
        return InteractionLog(config.log_path)
    except LogError as e:
        warning(f"Logging disabled: {e}")
        return None


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


def load_tools(config: Config, client: OllamaClient, asker: Asker | None = ask_question) -> ToolRegistry:
    tools = build_registry(config, client, asker)
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
    session.shares = ShareStore(config.shares_dir)
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


def start_session(config: Config, asker: Asker | None = ask_question) -> Session | None:
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
        agent.tools = load_tools(config, agent.client, asker)
        agent.guardrails = load_guardrails(config, agent.client)
    except (LLMError, GuardrailError, MemoryStoreError, MCPConfigError) as e:
        error(str(e))
        return None

    session = Session(config, agent, personas=personas, prompts=prompts, asker=asker)
    if config.features["rag"]:
        session.index = load_existing_index(config, agent.client)
        agent.retriever = session.index
    session.examples = agent.examples = load_examples(config, agent.client)
    return session
