import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from sensai.core.approval import Approval
from sensai.core.config import FEATURES, Config
from sensai.core.events import (
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
    SourcesEvent,
    ThoughtEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.export.report import FORMATS, ExportError, Publication, render, to_html
from sensai.core.export.shares import ShareError, public_link
from sensai.core.llm.ollama import ToolCall
from sensai.core.memory.conversation import ConversationMemory
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import cited_numbers
from sensai.core.tools.questions import Asker
from sensai.interfaces.cli.answer import (
    describe_call,
    describe_edits,
    guardrail_message,
    judgement_lines,
    name_conversation,
)
from sensai.interfaces.cli.commands.conversation import new_command, resume
from sensai.interfaces.cli.commands.document import report_of
from sensai.interfaces.cli.commands.settings import feature_command, feature_refusal
from sensai.interfaces.cli.commands.sharing import (
    blocked_message,
    publication_lines,
    publication_of,
    publish,
    share_refusal,
)
from sensai.interfaces.cli.output import plural
from sensai.interfaces.cli.session import ASK_USER, Session, save_session
from sensai.interfaces.cli.startup import load_log, load_memory, start_session
from sensai.interfaces.web.public import PAGE_HEADERS

STATIC_DIR = Path(__file__).resolve().parent / "static"
# The conversations hold the case and there is no login: only this machine may reach the server.
HOST = "127.0.0.1"
ALLOWED_HOSTS = [HOST, "localhost"]
POLL_SECONDS = 0.5
CHARS_PER_TOKEN = 4
# These change what the CLI commands do (/review, /cache, the scheduler): the page has no switch that would do nothing.
CLI_FEATURES = ("orchestration", "cache", "schedule")
Kind = Literal["conversation", "document"]


def context_used(memory: ConversationMemory) -> int:
    if memory.last_usage:
        return memory.last_usage.total_tokens
    # Ollama gave no count for a stopped answer or a resumed conversation: the gauge must not fall to zero.
    return sum(len(message["content"]) for message in memory.messages()) // CHARS_PER_TOKEN


def activity(text: str, warning: bool = False) -> dict:
    return {"type": "activity", "text": text, "warning": warning}


def cited(citations: list[str], answer: str) -> list[dict]:
    numbers = [number for number in sorted(cited_numbers(answer)) if 1 <= number <= len(citations)]
    return [{"number": number, "citation": citations[number - 1]} for number in numbers]


def page_event(event: Event, sources: list[SearchResult]) -> dict | None:
    """What the page shows of an event of the agent, None when it shows nothing."""
    if isinstance(event, TokenEvent):
        return {"type": "token", "text": event.text}
    if isinstance(event, DoneEvent):
        return {"type": "done", "sources": cited([source.citation for source in sources], event.content)}
    if isinstance(event, SourcesEvent):
        documents = len({source.source for source in event.sources})
        return activity(f"{plural(len(event.sources), 'excerpt')} from {plural(documents, 'document')}")
    if isinstance(event, ToolCallEvent):
        # The question of `ask_user` has its own card: it must not appear twice.
        return None if event.call.name == ASK_USER else activity(describe_call(event.call))
    if isinstance(event, ToolResultEvent):
        if event.result.error:
            return activity(f"{event.call.name} failed: {event.result.content.removeprefix('Error: ')}", warning=True)
        return activity(f"[{event.call.name}] {event.result.summary or 'done'}")
    if isinstance(event, GuardrailEvent):
        text = guardrail_message(event.hit)
        return activity(text, warning=event.hit.action != "redact") if text else None
    if isinstance(event, ThoughtEvent):
        return activity(f"Thought {event.number}: {event.thought}")
    if isinstance(event, ReasoningStoppedEvent):
        return activity(f"Reasoning stopped: {event.reason}. Sensai answers from what it found.", warning=True)
    if isinstance(event, ExamplesEvent):
        return activity(f"{plural(len(event.examples), 'example')} shown to the model")
    if isinstance(event, ExamplesFailedEvent):
        return activity(f"No examples were shown to the model: {event.message}", warning=True)
    if isinstance(event, DocumentEvent):
        return activity(f"Document: {describe_edits(event.report.applied)}") if event.report.applied else None
    if isinstance(event, DocumentFailedEvent):
        return activity(f"The document was not updated: {event.message}", warning=True)
    if isinstance(event, CompactionStartedEvent):
        return activity(f"Context at {event.tokens} tokens, condensing the oldest turns...")
    if isinstance(event, CompactionEvent):
        return activity(f"{plural(event.turns, 'turn')} condensed into the summary")
    if isinstance(event, CompactionFailedEvent):
        return activity(f"Could not condense the history: {event.message}", warning=True)
    if isinstance(event, CheckStartedEvent):
        sentences = plural(event.sentences, "sentence")
        return activity(f"Checking {sentences} with {event.model}, this can take a few minutes...")
    if isinstance(event, CheckEvent):
        lines = judgement_lines(event.judgement)
        return activity("\n".join(lines), warning=len(lines) > 1)
    if isinstance(event, CheckFailedEvent):
        return activity(f"The answer could not be checked: {event.message}", warning=True)
    if isinstance(event, LogFailedEvent):
        return activity(f"This message was not logged: {event.message}", warning=True)
    return None


class WebChat:
    """
    The conversation behind the page. Everything that touches the session runs on one thread: an SQLite connection
    only works on the thread that opened it. There is one conversation at a time, like in a terminal.
    """

    def __init__(self, open_session: Callable[[Asker], Session | None]):
        self.thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensai")
        self.session = self.call(open_session, self.ask)
        if self.session is not None:
            self.session.agent.approver = self.approve
        self.answering = threading.Lock()
        self.stopping = threading.Event()
        self.replies: queue.Queue[str | None] = queue.Queue()
        # What the page was asked and has not answered yet: "approval" or "question".
        self.asking: str | None = None
        self.events: queue.Queue[dict | None] = queue.Queue()

    def call(self, job: Callable, *args: Any) -> Any:
        return self.thread.submit(job, *args).result()

    def close(self) -> None:
        self.stop()
        if self.session is not None and self.session.index is not None:
            self.call(self.session.index.close)
        if self.session is not None and self.session.log is not None:
            self.call(self.session.log.close)
        self.thread.shutdown()

    def state(self) -> dict:
        session, config = self.session, self.session.config
        agent, memory = session.agent, session.agent.memory
        return {
            "model": config.model,
            "session": session.session_id,
            "persona": agent.persona.id if agent.persona else None,
            "personas": [
                {"id": persona.id, "name": persona.name, "role": persona.role} for persona in session.personas.values()
            ],
            "features": [
                {"name": name, "on": config.features[name], "description": feature.description}
                for name, feature in FEATURES.items() if name not in CLI_FEATURES
            ],
            "context": {
                "used": context_used(memory),
                "limit": config.num_ctx,
                "condensed_from": int(memory.limit),
            },
            "condensed": bool(memory.summary),
            "document": not memory.document.is_empty,
            "sharing": config.features["sharing"],
            "turns": [
                {
                    "user": turn.user, "assistant": turn.assistant, "sources": cited(turn.citations, turn.assistant),
                    "tools": [f"[{tool.name}] {tool.result}" for tool in turn.tools],
                }
                for turn in memory.turns
            ],
            "sessions": [
                {"id": info.id, "title": info.title, "updated": f"{info.updated:%Y-%m-%d %H:%M}"}
                for info in session.store.list()
            ] if config.features["sessions"] else [],
        }

    def start(self, message: str) -> queue.Queue | None:
        """Starts the answer on the thread of the session. None when another answer is running."""
        if not self.answering.acquire(blocking=False):
            return None
        self.stopping.clear()
        self.events = queue.Queue()
        self.thread.submit(self._answer, message)
        return self.events

    def stop(self) -> None:
        """Takes effect at the next event of the agent: a call to the model or a tool cannot be cut from outside."""
        self.stopping.set()
        self.replies.put(None)

    def _answer(self, message: str) -> None:
        try:
            try:
                self._stream(message)
            except Exception as e:
                # Nobody is there to restart the server: the page shows the error and the next message still works.
                self.events.put({"type": "error", "text": str(e) or type(e).__name__})
            save_session(self.session)
            self.events.put({"type": "state", **self.state()})
        finally:
            self.answering.release()
            self.events.put(None)

    def _stream(self, message: str) -> None:
        name_conversation(self.session)
        agent = self.session.agent
        events = agent.run(message)
        sources: list[SearchResult] = []
        for event in events:
            if self.stopping.is_set():
                # Closes the connection too: Ollama stops generating.
                events.close()
                kept = self.session.config.features["branching"] and agent.interrupt()
                self.events.put({"type": "stopped", "kept": kept})
                return
            if isinstance(event, SourcesEvent):
                sources = event.sources
            shown = page_event(event, sources)
            if shown is not None:
                self.events.put(shown)

    def wait_for_page(self, event: dict) -> str | None:
        """Blocks the thread of the session until the click. None when the answer is stopped or the page closed."""
        # A click that came too late, or the stop of the last answer, must not answer this question.
        while not self.replies.empty():
            self.replies.get_nowait()
        if self.stopping.is_set():
            return None
        self.asking = event["type"]
        self.events.put(event)
        reply = self.replies.get()
        self.asking = None
        return reply

    def reply(self, kind: str, value: str | None) -> bool:
        if self.asking != kind:
            return False
        self.replies.put(value)
        return True

    def approve(self, call: ToolCall, reason: str) -> Approval:
        if call.name in self.session.always_allowed:
            return Approval(True)
        # The exact call was shown just before, by the ToolCallEvent.
        reason = f"{reason[:1].upper()}{reason[1:]}."
        choice = self.wait_for_page({"type": "approval", "tool": call.name, "reason": reason})
        if choice is None:
            return Approval(False, "the user stopped the answer")
        if choice == "always":
            self.session.always_allowed.add(call.name)
        return Approval(choice != "deny")

    def ask(self, question: str, options: list[str]) -> str | None:
        """Asks the question of the model on the page. None tells the model that nobody answered."""
        return self.wait_for_page({"type": "question", "question": question, "options": options})

    def new(self) -> None:
        new_command(self.session)

    def resume(self, session_id: str) -> bool:
        return resume(self.session, session_id)

    def set_persona(self, persona_id: str | None) -> bool:
        session, agent = self.session, self.session.agent
        if persona_id is None:
            agent.persona, agent.memory.system_prompt = None, session.config.system_prompt
            active = session.prompts.active() if session.prompts and session.config.features["prompt_versions"] else None
            if active:
                agent.prompt_version, agent.memory.system_prompt = active.number, active.prompt
            return True
        persona = session.personas.get(persona_id)
        if persona is None:
            return False
        agent.set_persona(persona)
        return True

    def set_feature(self, name: str, on: bool) -> None:
        value = "on" if on else "off"
        feature_command(self.session, f"{name} {value}")

    def export(self, kind: str, suffix: str) -> bytes:
        memory = self.session.agent.memory
        if kind == "conversation" and not (memory.turns or memory.summary):
            raise ExportError("No message to export yet.")
        return render(report_of(self.session, kind), suffix)

    def public_links(self) -> list[dict]:
        return [
            {
                "token": share.token, "kind": share.kind, "title": share.title,
                "created": f"{share.created:%Y-%m-%d %H:%M}", "link": public_link(self.session.config, share),
            }
            for share in self.session.shares.list()
        ]

    def _publication(self, kind: str) -> Publication:
        refusal = share_refusal(self.session, kind)
        if refusal:
            raise ShareError(refusal)
        return publication_of(self.session, kind)

    def publication(self, kind: str) -> dict:
        publication = self._publication(kind)
        blocked = blocked_message(publication.blocked) if publication.blocked else None
        return {"lines": publication_lines(self.session, publication), "blocked": blocked}

    def publication_page(self, kind: str) -> str:
        return to_html(self._publication(kind).report)

    def publish(self, kind: str) -> dict:
        publication = self._publication(kind)
        if publication.blocked:
            raise ShareError(blocked_message(publication.blocked))
        return {"link": public_link(self.session.config, publish(self.session, kind, publication))}


class Message(BaseModel):
    message: str


class ApprovalChoice(BaseModel):
    choice: Literal["allow", "always", "deny"]


class QuestionAnswer(BaseModel):
    answer: str | None


class PersonaChoice(BaseModel):
    id: str | None


class FeatureState(BaseModel):
    on: bool


def same_origin(request: Request) -> None:
    # Any site open in the browser can send requests to localhost: only the page itself is listened to.
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "Requests from another site are refused")


async def stream(chat: WebChat, events: queue.Queue, request: Request) -> AsyncIterator[str]:
    try:
        while True:
            try:
                event = await asyncio.to_thread(events.get, timeout=POLL_SECONDS)
            except queue.Empty:
                if await request.is_disconnected():
                    return
                continue
            if event is None:
                return
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        # Also reached when the page is closed: an answer nobody reads, or an approval nobody can give, stops there.
        chat.stop()


def create_app(chat: WebChat) -> FastAPI:
    app = FastAPI(title="Sensai", docs_url=None, redoc_url=None, openapi_url=None, dependencies=[Depends(same_origin)])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.get("/api/state")
    def state() -> dict:
        return chat.call(chat.state)

    @app.post("/api/chat")
    async def answer(body: Message, request: Request) -> StreamingResponse:
        events = chat.start(body.message)
        if events is None:
            raise HTTPException(409, "An answer is already running")
        return StreamingResponse(
            stream(chat, events, request), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/stop")
    def stop() -> dict:
        chat.stop()
        return {}

    @app.post("/api/approval")
    def approval(body: ApprovalChoice) -> dict:
        if not chat.reply("approval", body.choice):
            raise HTTPException(409, "No tool call is waiting for an answer")
        return {}

    @app.post("/api/question")
    def question(body: QuestionAnswer) -> dict:
        if not chat.reply("question", body.answer):
            raise HTTPException(409, "No question is waiting for an answer")
        return {}

    @app.post("/api/new")
    def new() -> dict:
        chat.call(chat.new)
        return chat.call(chat.state)

    @app.post("/api/sessions/{session_id}")
    def resume_session(session_id: str) -> dict:
        if not chat.call(chat.resume, session_id):
            raise HTTPException(404, f"Cannot open the conversation {session_id}")
        return chat.call(chat.state)

    @app.post("/api/persona")
    def persona(body: PersonaChoice) -> dict:
        if not chat.call(chat.set_persona, body.id):
            raise HTTPException(404, f"Unknown persona '{body.id}'")
        return chat.call(chat.state)

    @app.post("/api/features/{name}")
    def feature(name: str, body: FeatureState) -> dict:
        if name not in FEATURES or name in CLI_FEATURES:
            raise HTTPException(404, f"Unknown feature '{name}'")
        refusal = feature_refusal(chat.session.config, name, "on" if body.on else "off")
        if refusal:
            raise HTTPException(400, refusal)
        chat.call(chat.set_feature, name, body.on)
        return chat.call(chat.state)

    def while_idle(job: Callable, *args: Any) -> Any:
        # The thread of the session is taken during an answer, and it may be waiting for a click on this very page.
        if chat.answering.locked():
            raise HTTPException(409, "An answer is running: wait for its end or stop it")
        try:
            return chat.call(job, *args)
        except (ExportError, ShareError) as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/export/{kind}.{suffix}")
    def export(kind: Kind, suffix: Literal["md", "json", "pdf", "html"]) -> Response:
        name = "-".join(filter(None, ["sensai", kind, chat.session.session_id]))
        headers = {"Content-Disposition": f'attachment; filename="{name}.{suffix}"'}
        return Response(while_idle(chat.export, kind, f".{suffix}"), media_type=FORMATS[f".{suffix}"], headers=headers)

    # The links are files: they do not wait for the answer, and taking one down never should.
    @app.get("/api/shares")
    def shares() -> dict:
        return {"shares": chat.public_links()}

    @app.get("/api/shares/{kind}")
    def publication(kind: Kind) -> dict:
        return while_idle(chat.publication, kind)

    @app.get("/api/shares/{kind}/page")
    def publication_page(kind: Kind) -> HTMLResponse:
        return HTMLResponse(while_idle(chat.publication_page, kind), headers=PAGE_HEADERS)

    @app.post("/api/shares/{kind}")
    def publish_page(kind: Kind) -> dict:
        return while_idle(chat.publish, kind)

    @app.delete("/api/shares/{token}")
    def remove_share(token: str) -> dict:
        try:
            removed = chat.session.shares.remove(token)
        except ShareError as e:
            raise HTTPException(500, str(e)) from None
        if not removed:
            raise HTTPException(404, "This link does not exist")
        return {}

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True))
    return app


def open_session(config: Config, asker: Asker) -> Session | None:
    session = start_session(config, asker)
    if session is None:
        return None
    if config.features["logging"]:
        session.log = session.agent.log = load_log(config)
    load_memory(session, None)
    return session


def run_web(config: Config, port: int) -> int:
    chat = WebChat(lambda asker: open_session(config, asker))
    if chat.session is None:
        return 1
    print(f"Sensai web UI: http://{HOST}:{port} (Ctrl+C to stop)", flush=True)
    try:
        uvicorn.run(create_app(chat), host=HOST, port=port, log_level="warning")
    finally:
        chat.close()
    return 0
