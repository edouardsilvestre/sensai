from pathlib import Path

from sensai.core.config import API_KEY_HELP, APPROVAL_MODES, FEATURES, Config
from sensai.core.guardrails.rules import STAGES
from sensai.core.tools.permissions import ALWAYS_DENIED, Permissions
from sensai.interfaces.cli.output import error
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import (
    guardrails_source,
    load_cache,
    load_examples,
    load_existing_index,
    load_log,
    load_prompts,
    load_schedule,
    reload_tools,
)

APPROVAL_DESCRIPTIONS = {
    "critical": "asks before the tools that send data out of the machine or run code",
    "all": "asks before every tool call",
    "off": "never asks",
}


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


def feature_refusal(config: Config, name: str, value: str) -> str | None:
    if name == "web_search" and value == "on" and not config.ollama_api_key:
        return f"web_search needs OLLAMA_API_KEY in .env: {API_KEY_HELP}"
    return None


def feature_command(session: Session, argument: str) -> None:
    name, _, value = argument.lower().partition(" ")
    value = value.strip()
    if name not in FEATURES or value not in ("on", "off"):
        error(f"Usage: /feature NAME on|off, with NAME among: {', '.join(FEATURES)}")
        return
    config = session.config
    refusal = feature_refusal(config, name, value)
    if refusal:
        error(refusal)
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
    if name == "logging" and value == "on" and session.log is None:
        session.log = session.agent.log = load_log(config)
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
