import asyncio
import json
import queue
import threading
import time

import pytest
from fastapi.testclient import TestClient

from conftest import FakeClient
from sensai.core.agent import INTERRUPTED, Agent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.memory.conversation import Turn
from sensai.core.memory.sessions import SessionStore
from sensai.core.persona import load_personas
from sensai.core.rag.models import SearchResult
from sensai.core.tools.registry import ToolRegistry, requires_approval
from sensai.interfaces.cli.main import Session, load_memory, parse_args
from sensai.interfaces.web.server import HOST, STATIC_DIR, WebChat, create_app, stream

SENT: list[str] = []
THREADS: set[int] = set()


@requires_approval("the query leaves the machine")
def send(query: str) -> str:
    """
    Send a query.

    Args:
      query: The query

    Returns:
      The result
    """
    SENT.append(query)
    THREADS.add(threading.get_ident())
    return f"sent {query}"


class Excerpts:
    def search(self, query: str, k: int) -> list[SearchResult]:
        return [SearchResult(chunk_id=1, text="Orion", location="p. 1", source="statutes.pdf", score=0.9, description="")]


class SlowClient(FakeClient):
    """Keeps writing until the test has seen it start."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()

    def chat_stream(self, *args, **kwargs):
        for number in range(500):
            yield f"word{number} "
            self.started.set()
            time.sleep(0.01)


class DownClient(FakeClient):
    def chat_stream(self, *args, **kwargs):
        raise LLMError("Ollama is not running")
        yield


def start(config, client, **agent) -> WebChat:
    def open_session() -> Session:
        tools = ToolRegistry([send])
        session = Session(
            config, Agent(config, client, tools=tools, **agent), personas=load_personas(config.personas_dir)
        )
        load_memory(session, None)
        return session

    return WebChat(open_session)


@pytest.fixture(autouse=True)
def nothing_sent():
    SENT.clear()
    THREADS.clear()


@pytest.fixture
def chat(config, client):
    chat = start(config, client)
    yield chat
    chat.close()


def page_of(chat: WebChat) -> TestClient:
    return TestClient(create_app(chat), base_url=f"http://{HOST}")


@pytest.fixture
def page(chat):
    return page_of(chat)


def events_of(response) -> list[dict]:
    return [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]


def ask(page: TestClient, message: str = "Who approved the transfer?") -> list[dict]:
    return events_of(page.post("/api/chat", json={"message": message}))


def of_type(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["type"] == kind]


def in_background(page: TestClient, message: str = "Who?") -> tuple[threading.Thread, dict]:
    """Sends the message from another thread: the test has to act while the answer runs."""
    result: dict = {}

    def run() -> None:
        result["events"] = ask(page, message)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def wait_for(condition) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_the_page_is_served(page):
    html = page.get("/").text
    assert "<title>Sensai</title>" in html and 'src="app.js"' in html
    assert page.get("/app.js").status_code == 200 and page.get("/style.css").status_code == 200


def test_the_page_loads_nothing_from_the_internet():
    for name in ("index.html", "style.css", "app.js"):
        text = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert "https://" not in text and "//cdn" not in text and "@import" not in text
    assert "http://" not in (STATIC_DIR / "app.js").read_text(encoding="utf-8")


def test_the_answers_are_never_inserted_as_html_without_escaping():
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert script.count("innerHTML") == 2 and script.count("innerHTML = markdown(") == 2


def test_the_state_describes_the_page(page, config):
    state = page.get("/api/state").json()
    assert state["model"] == config.model and state["session"] is None and state["turns"] == []
    assert state["context"] == {"used": 0, "limit": 8192, "condensed_from": 6144}
    assert [persona["id"] for persona in state["personas"]] == ["editor", "fact_checker", "investigator"]
    assert {"name": "rag", "on": True, "description": "answer from the indexed documents"} in state["features"]


def test_an_answer_is_streamed_then_saved(page, chat):
    events = ask(page)
    assert "".join(event["text"] for event in of_type(events, "token")) == "Hello there "
    assert [event["type"] for event in events[-2:]] == ["done", "state"]
    state = events[-1]
    assert state["turns"] == [
        {"user": "Who approved the transfer?", "assistant": "Hello there ", "sources": [], "tools": []}
    ]
    assert state["context"]["used"] == 110
    assert [session["id"] for session in state["sessions"]] == [state["session"]]
    assert state["sessions"][0]["title"] == "Who approved the transfer?"


def test_the_next_message_has_the_history(page, client):
    ask(page, "First")
    ask(page, "Second")
    assert [message["content"] for message in client.chats[-1][0]] == ["First", "Hello there ", "Second"]


def test_the_sources_cited_are_listed(config):
    chat = start(config, FakeClient("Orion paid [1] and more [7]."), retriever=Excerpts())
    try:
        events = ask(page_of(chat))
    finally:
        chat.close()
    assert of_type(events, "activity")[0]["text"] == "1 excerpt from 1 document"
    expected = [{"number": 1, "citation": "statutes.pdf, p. 1"}]
    assert of_type(events, "done")[0]["sources"] == expected
    assert events[-1]["turns"][0]["sources"] == expected


def test_an_empty_message_is_reported(page):
    events = ask(page, "   ")
    assert of_type(events, "error")[0]["text"] == "Empty message"
    assert of_type(ask(page), "done")


def test_a_failure_of_the_model_is_shown_and_the_next_message_works(config, client, monkeypatch):
    chat = start(config, client)
    page = page_of(chat)
    try:
        monkeypatch.setattr(client, "chat_stream", DownClient().chat_stream)
        assert of_type(ask(page), "error")[0]["text"] == "Ollama is not running"
        monkeypatch.undo()
        assert of_type(ask(page), "done")
    finally:
        chat.close()


def test_an_unexpected_error_does_not_stop_the_server(page, client, monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError
        yield

    monkeypatch.setattr(client, "chat_stream", crash)
    assert of_type(ask(page), "error")[0]["text"] == "RuntimeError"
    assert page.get("/api/state").status_code == 200


def test_a_second_answer_is_refused_while_one_is_running(page, chat):
    chat.answering.acquire()
    response = page.post("/api/chat", json={"message": "Hi"})
    chat.answering.release()
    assert response.status_code == 409


def test_a_tool_waits_for_the_click_and_shows_the_exact_call(page, chat, client):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    thread, result = in_background(page)
    wait_for(lambda: chat.asking)
    assert SENT == []
    assert page.post("/api/approval", json={"choice": "allow"}).status_code == 200
    thread.join(5)
    kinds = [event["type"] for event in result["events"]]
    assert of_type(result["events"], "activity")[0]["text"] == '[send] query="Orion"'
    assert kinds.index("activity") < kinds.index("approval")
    assert of_type(result["events"], "approval")[0]["reason"] == "The query leaves the machine."
    assert SENT == ["Orion"]


def test_a_saved_conversation_shows_its_tool_calls(page, chat, client):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    thread, result = in_background(page)
    wait_for(lambda: chat.asking)
    page.post("/api/approval", json={"choice": "allow"})
    thread.join(5)
    assert result["events"][-1]["turns"][0]["tools"] == ["[send] sent Orion"]


def test_a_denied_tool_does_not_run(page, chat, client):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    thread, result = in_background(page)
    wait_for(lambda: chat.asking)
    page.post("/api/approval", json={"choice": "deny"})
    thread.join(5)
    assert SENT == [] and of_type(result["events"], "done")


def test_always_does_not_ask_again(page, chat, client):
    client.tool_rounds = [[ToolCall("send", {"query": "one"})]]
    thread, _ = in_background(page)
    wait_for(lambda: chat.asking)
    page.post("/api/approval", json={"choice": "always"})
    thread.join(5)
    client.tool_rounds = [[ToolCall("send", {"query": "two"})]]
    assert not of_type(ask(page), "approval")
    assert SENT == ["one", "two"]


def test_an_approval_nobody_asked_for_is_refused(page):
    assert page.post("/api/approval", json={"choice": "allow"}).status_code == 409
    assert page.post("/api/approval", json={"choice": "maybe"}).status_code == 422


def test_a_late_click_does_not_answer_the_next_question(page, chat, client):
    chat.approvals.put("allow")
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    thread, _ = in_background(page)
    wait_for(lambda: chat.asking)
    assert SENT == []
    page.post("/api/approval", json={"choice": "deny"})
    thread.join(5)
    assert SENT == []


def test_stop_keeps_what_was_written(config):
    client = SlowClient()
    chat = start(config, client)
    page = page_of(chat)
    try:
        thread, result = in_background(page)
        assert client.started.wait(5)
        page.post("/api/stop")
        thread.join(5)
        kept = chat.session.agent.memory.turns[0].assistant
        assert of_type(result["events"], "stopped") == [{"type": "stopped", "kept": True}]
        assert kept.startswith("word0") and kept.endswith(INTERRUPTED) and "word499" not in kept
        assert result["events"][-1]["turns"][0]["assistant"] == kept
        assert SessionStore(config.sessions_dir).list()[0].turns == 1
    finally:
        chat.close()


def test_the_gauge_does_not_fall_to_zero_after_a_stop_or_a_resume(page, chat, config):
    memory = chat.session.agent.memory
    memory.turns = [Turn("a" * 400, "b" * 400)]
    SessionStore(config.sessions_dir).save("long", memory, "m")
    assert page.post("/api/sessions/long").json()["context"]["used"] == 200


def test_stop_keeps_nothing_when_branching_is_off(config):
    config.features["branching"] = False
    client = SlowClient()
    chat = start(config, client)
    page = page_of(chat)
    try:
        thread, result = in_background(page)
        assert client.started.wait(5)
        page.post("/api/stop")
        thread.join(5)
        assert of_type(result["events"], "stopped") == [{"type": "stopped", "kept": False}]
        assert chat.session.agent.memory.turns == []
    finally:
        chat.close()


def test_stop_during_an_approval_does_not_run_the_tool(page, chat, client):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    thread, result = in_background(page)
    wait_for(lambda: chat.asking)
    page.post("/api/stop")
    thread.join(5)
    assert SENT == [] and of_type(result["events"], "stopped")


class LeftPage:
    async def is_disconnected(self) -> bool:
        return True


def test_a_page_that_leaves_stops_the_answer(config, monkeypatch):
    monkeypatch.setattr("sensai.interfaces.web.server.POLL_SECONDS", 0.01)
    chat = start(config, FakeClient())

    async def read() -> list[str]:
        return [item async for item in stream(chat, queue.Queue(), LeftPage())]

    try:
        assert asyncio.run(read()) == [] and chat.stopping.is_set()
    finally:
        chat.close()


def test_a_page_closed_during_an_approval_refuses_the_tool(page, chat, client):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    events = chat.start("Who?")
    wait_for(lambda: chat.asking)
    chat.stop()
    wait_for(lambda: not chat.asking)
    while events.get(timeout=5) is not None:
        pass
    assert SENT == []


def test_everything_runs_on_the_thread_that_opened_the_session(config, client):
    """An SQLite connection refuses any other thread."""
    opened = []
    chat = WebChat(lambda: opened.append(threading.get_ident()) or Session(config, Agent(config, client)))
    try:
        assert chat.call(threading.get_ident) == opened[0]
    finally:
        chat.close()


def test_the_tools_run_on_the_thread_of_the_session(page, chat, client):
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    thread, _ = in_background(page)
    wait_for(lambda: chat.asking)
    page.post("/api/approval", json={"choice": "allow"})
    thread.join(5)
    assert THREADS == {chat.call(threading.get_ident)}


def test_new_chat_keeps_the_previous_one(page):
    previous = ask(page)[-1]["session"]
    state = page.post("/api/new").json()
    assert state["turns"] == [] and state["session"] is None
    assert [session["id"] for session in state["sessions"]] == [previous]


def test_a_saved_conversation_is_opened_and_continued(page, chat, client, config):
    memory = chat.session.agent.memory
    memory.turns = [Turn("Who is Marc Vidal?", "The CFO.")]
    SessionStore(config.sessions_dir).save("vidal", memory, "m")
    page.post("/api/new")
    state = page.post("/api/sessions/vidal").json()
    assert state["session"] == "vidal" and state["turns"][0]["assistant"] == "The CFO."
    ask(page, "And his boss?")
    assert client.chats[-1][0][0]["content"] == "Who is Marc Vidal?"


@pytest.mark.parametrize("session_id", ["nope", "not.an.id"])
def test_an_unknown_conversation_is_refused(page, session_id):
    ask(page)
    assert page.post(f"/api/sessions/{session_id}").status_code == 404
    assert len(page.get("/api/state").json()["turns"]) == 1


def test_a_path_is_not_a_conversation(page):
    assert page.post("/api/sessions/..%2F..%2Fsecret").status_code == 405


def test_the_persona_changes_and_the_history_stays(page, client):
    ask(page)
    state = page.post("/api/persona", json={"id": "editor"}).json()
    assert state["persona"] == "editor" and len(state["turns"]) == 1
    ask(page, "Next")
    assert client.chats[-1][0][0]["role"] == "system" and "editor" in client.chats[-1][0][0]["content"].lower()


def test_default_goes_back_to_no_persona(page, chat, config):
    config.system_prompt = "You help journalists."
    page.post("/api/persona", json={"id": "editor"})
    state = page.post("/api/persona", json={"id": None}).json()
    assert state["persona"] is None and chat.session.agent.memory.system_prompt == "You help journalists."


def test_an_unknown_persona_is_refused(page):
    assert page.post("/api/persona", json={"id": "nope"}).status_code == 404


def test_a_feature_is_turned_off(page, chat):
    state = page.post("/api/features/document", json={"on": False}).json()
    assert {"name": "document", "on": False} .items() <= next(
        feature for feature in state["features"] if feature["name"] == "document"
    ).items()
    assert chat.session.config.features["document"] is False


def test_web_search_without_a_key_is_refused_with_the_reason(page, chat):
    response = page.post("/api/features/web_search", json={"on": True})
    assert response.status_code == 400 and "OLLAMA_API_KEY" in response.json()["detail"]
    assert chat.session.config.features["web_search"] is False


@pytest.mark.parametrize("name", ["nope", "cache", "orchestration", "schedule"])
def test_a_feature_the_page_does_not_use_is_refused(page, name):
    assert page.post(f"/api/features/{name}", json={"on": True}).status_code == 404
    assert name not in [feature["name"] for feature in page.get("/api/state").json()["features"]]


def test_the_guardrails_apply_and_are_shown(config):
    rules = [{"name": "source", "terms": ["Claire Benoit"], "on": ["output"], "action": "redact"}]
    chat = start(config, FakeClient("The source is Claire Benoit."), guardrails=Guardrails(parse_rules(rules)))
    try:
        events = ask(page_of(chat))
    finally:
        chat.close()
    assert "Claire Benoit" not in json.dumps(events)
    assert any("source" in event["text"] and "redacted" in event["text"] for event in of_type(events, "activity"))


def test_a_blocked_message_reaches_nobody(config, client):
    rules = [{"name": "source", "terms": ["Claire"], "on": ["input"], "action": "block"}]
    chat = start(config, client, guardrails=Guardrails(parse_rules(rules)))
    try:
        events = ask(page_of(chat), "Where is Claire?")
    finally:
        chat.close()
    assert of_type(events, "activity")[0]["warning"] is True
    assert client.chats == [] and events[-1]["turns"] == []


def test_a_request_from_another_site_is_refused(page):
    headers = {"Origin": "https://evil.example"}
    assert page.post("/api/new", headers=headers).status_code == 403
    assert page.post("/api/chat", json={"message": "Hi"}, headers=headers).status_code == 403
    assert page.post("/api/new", headers={"Origin": f"http://{HOST}"}).status_code == 200


def test_another_host_name_is_refused(chat):
    """A DNS rebinding attack reaches 127.0.0.1 under the name of another site."""
    assert TestClient(create_app(chat), base_url="http://evil.example").get("/api/state").status_code == 400


def test_the_server_only_listens_on_this_machine():
    assert HOST == "127.0.0.1"


def test_the_web_command():
    args = parse_args(["web", "--port", "9000"])
    assert args.command == "web" and args.port == 9000
    assert parse_args(["web"]).port == 8000
