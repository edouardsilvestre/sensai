import json
import textwrap
from collections.abc import Callable, Generator, Iterator

from sensai.core.agent import EmptyInputError
from sensai.core.config import Config
from sensai.core.document.document import Edit, EditReport
from sensai.core.evaluation.judge import UNCHECKED, Claim, Judgement
from sensai.core.events import (
    AgentStartedEvent,
    CacheFailedEvent,
    CacheHitEvent,
    CheckEvent,
    CheckFailedEvent,
    CheckStartedEvent,
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
    LogFailedEvent,
    ReasoningStoppedEvent,
    RerankEvent,
    RerankFailedEvent,
    ReviewDoneEvent,
    SourcesEvent,
    ThoughtEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.guardrails.rules import Hit
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers
from sensai.core.rag.store import StoreError
from sensai.core.structured import StructuredError, cited_sources, parse_answer
from sensai.interfaces.cli.output import error, plural, warning
from sensai.interfaces.cli.session import ASK_USER, Session, save_session


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


def rerank_report(event: RerankEvent, config: Config) -> str:
    parts = [f"{event.kept} kept"]
    # Relevant enough, but past the SENSAI_RAG_TOP_K best.
    left_out = event.candidates - event.kept - len(event.dropped)
    if left_out:
        parts.append(f"{left_out} left out after the {config.rag_top_k} best")
    if event.dropped:
        parts.append(f"{len(event.dropped)} dropped under {config.rerank_threshold}% relevance (/sources dropped)")
    return f"[rerank] {plural(event.candidates, 'candidate')}: " + ", ".join(parts)


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


def guardrail_message(hit: Hit) -> str | None:
    where, found = WHERE[hit.stage], f" ({hit.found})" if hit.found else ""
    if hit.action == "redact":
        return f"[guardrail] {hit.rule}: {plural(hit.count, 'passage')} redacted in {where}{found}"
    if hit.action == "warn":
        return f"guardrail {hit.rule}: found in {where}{found}" + (f". {hit.message}" if hit.message else "")
    if hit.stage == "input":
        return f"Message blocked by the guardrail {hit.rule}: {hit.message or hit.found}. Nothing was sent to the model."
    if hit.stage == "output":
        return f"Answer stopped by the guardrail {hit.rule}{found}" + (f": {hit.message}" if hit.message else "")
    # A blocked tool call is reported by its result, which the model also gets.
    return None


def print_guardrail(hit: Hit) -> None:
    message = guardrail_message(hit)
    if message is None:
        return
    if hit.action == "redact":
        print(message)
    else:
        warning(message)


def claim_label(claim: Claim) -> str:
    if claim.verdict == UNCHECKED:
        return "unchecked"
    if claim.by == "code":
        return "not in the excerpts"
    return "contradicted" if claim.verdict == "contradicted" else "to verify"


def judgement_lines(judgement: Judgement) -> list[str]:
    supported = sum(claim.verdict == "supported" for claim in judgement.claims)
    faithfulness = "-" if judgement.faithfulness is None else f"{judgement.faithfulness:.0%}"
    scores = ", ".join(
        f"{name} {'-' if score is None else score}/5"
        for name, score in (("relevance", judgement.relevance), ("coherence", judgement.coherence))
    )
    lines = [
        f"[check] {plural(len(judgement.claims), 'sentence')}: {supported} supported, "
        f"faithfulness {faithfulness}, {scores}"
    ]
    for claim in judgement.claims:
        if claim.verdict != "supported":
            lines.append(f'  {claim_label(claim)}: "{claim.text}"' + (f" ({claim.reason})" if claim.reason else ""))
    return lines


def print_judgement(judgement: Judgement) -> None:
    print("\n".join(judgement_lines(judgement)))


def chat_turn(session: Session, line: str) -> None:
    if line.strip():
        session.last_sources = []
        session.last_dropped = []
        session.examples_used = []
    turns = session.agent.memory.turns
    last = turns[-1] if turns else None
    show_events(session, session.agent.run(line))
    turns = session.agent.memory.turns
    if turns and turns[-1] is not last:
        session.answered = (turns[-1], list(session.last_sources))


def name_conversation(session: Session) -> None:
    agent = session.agent
    if session.session_id is not None:
        agent.session_id = session.session_id
    elif not agent.session_id and session.store is not None:
        agent.session_id = session.store.new_id()


def show_events(
    session: Session,
    events: Generator[Event, None, None] | Iterator[Event],
    structured: bool = False,
    interrupt: Callable[[], bool] | None = None,
) -> None:
    name_conversation(session)
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
            elif isinstance(event, LogFailedEvent):
                warning(f"This message was not logged: {event.message}")
            elif isinstance(event, CheckStartedEvent):
                print(
                    f"[check] Checking {plural(event.sentences, 'sentence')} with {event.model}, "
                    "this can take a few minutes (Ctrl+C to skip)..."
                )
            elif isinstance(event, CheckEvent):
                print_judgement(event.judgement)
            elif isinstance(event, CheckFailedEvent):
                warning(f"The answer could not be checked: {event.message}")
            elif isinstance(event, ReviewDoneEvent):
                rounds = plural(event.rounds, "check")
                if event.approved:
                    print(f"\n[review] approved by the fact-checker after {rounds}, the editor's text is kept.")
                else:
                    warning(f"Answer not approved after {rounds}: the last answer is kept, check the remarks above.")
            elif isinstance(event, RerankEvent):
                session.last_dropped = event.dropped
                print(rerank_report(event, session.config))
            elif isinstance(event, RerankFailedEvent):
                warning(f"Excerpts not reranked until Sensai restarts: {event.message}")
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
                if event.call.name != ASK_USER:
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


def print_structured(answer: str, sources: list[SearchResult]) -> None:
    try:
        data = parse_answer(answer)
    except StructuredError as e:
        error(f"{e}:\n{answer}")
        return
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if sources:
        print_sources(sources, cited_sources(data))
