import shlex
import textwrap
from pathlib import Path

from sensai.core.llm.ollama import LLMError
from sensai.core.rag.models import SearchFilter, SearchResult, parse_filter
from sensai.core.rag.store import StoreError
from sensai.interfaces.cli.output import error
from sensai.interfaces.cli.runs.index import ingest
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import open_index


def describe_score(source: SearchResult) -> str:
    if source.relevance is None:
        return f"similarity {source.score:.2f}"
    return f"relevance {source.relevance:.2f}, similarity {source.score:.2f}"


def show_dropped(dropped: list[SearchResult]) -> None:
    if not dropped:
        print("No excerpt was dropped for the last message.")
        return
    # Not numbered like the sources: the answer cannot cite them.
    for source in dropped:
        print(f"\n- {source.citation} ({describe_score(source)})")
        print(textwrap.indent(source.text, "    "))


def show_excerpts(sources: list[SearchResult]) -> None:
    if not sources:
        print("No excerpts were retrieved for the last message.")
        return
    for number, source in enumerate(sources, start=1):
        print(f"\n[{number}] {source.citation} ({describe_score(source)})")
        print(textwrap.indent(source.text, "    "))


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


FILTER_USAGE = "Usage: /filter [clear | after:2023-03 before:2023-07 from:NAME to:NAME kind:KIND source:FOLDER/]"


def filter_command(session: Session, argument: str) -> None:
    if argument.lower() == "clear":
        session.agent.search_filter = SearchFilter()
    elif argument:
        try:
            session.agent.search_filter = parse_filter(argument)
        except ValueError as e:
            error(f"{e}\n{FILTER_USAGE}")
            return
    search_filter = session.agent.search_filter
    if search_filter.is_empty():
        print("No filter: the searches look at every document.")
        return
    matching = ""
    if session.index is not None:
        count = session.index.count_documents([search_filter])
        matching = f": the searches look at {count} of the {session.index.stats().documents} indexed documents"
    print(f"Filter: {search_filter.describe()}{matching}.")
