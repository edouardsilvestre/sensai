import textwrap
from datetime import datetime

from sensai.core.agent import INTERRUPTED, STOPPED
from sensai.core.examples import Example, ExampleError, make_example
from sensai.core.monitoring.log import LogError
from sensai.interfaces.cli.answer import describe_call
from sensai.interfaces.cli.output import error, plural
from sensai.interfaces.cli.runs.stats import logged_since, print_stats
from sensai.interfaces.cli.session import Session


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


def stats_command(session: Session, argument: str) -> None:
    if argument and argument != "clear" and not (argument.isdecimal() and int(argument) >= 1):
        error("Usage: /stats [DAYS|clear], with DAYS at least 1")
        return
    log = session.log
    if log is None:
        if session.config.features["logging"]:
            print("The log could not be opened: /feature logging on to try again.")
        else:
            print("The logging feature is off: /feature logging on to log the messages and see their metrics.")
        return
    try:
        if argument == "clear":
            print(f"Log cleared: {plural(log.clear(), 'message')} removed.")
            return
        days = int(argument or 7)
        print_stats(logged_since(log, days), days)
    except LogError as e:
        error(str(e))
