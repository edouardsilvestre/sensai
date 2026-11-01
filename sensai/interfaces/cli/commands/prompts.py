import textwrap
from datetime import datetime

from sensai.core.llm.ollama import LLMError
from sensai.core.prompts import PromptError, PromptVersion, VersionScore, compare, comparison, diff
from sensai.interfaces.cli.output import error, plural, warning
from sensai.interfaces.cli.session import Session


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
