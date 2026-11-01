import asyncio
import json
import queue
import threading
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from conftest import FakeClient
from sensai.core.agent import INTERRUPTED, Agent
from sensai.core.document.document import Edit
from sensai.core.evaluation.judge import Claim, Judgement
from sensai.core.events import CheckEvent, CheckFailedEvent, CheckStartedEvent, LogFailedEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.memory.conversation import Turn
from sensai.core.memory.sessions import SessionStore
from sensai.core.monitoring.log import InteractionLog
from sensai.core.persona import load_personas
from sensai.core.prompts import PromptStore
from sensai.core.rag.models import SearchResult
from sensai.core.tools.questions import NO_ANSWER, Questions
from sensai.core.tools.registry import ToolRegistry, requires_approval
from sensai.interfaces.cli.main import parse_args
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import load_memory
from sensai.interfaces.web.server import HOST, STATIC_DIR, WebChat, create_app, open_session, page_event, stream

SENT: list[str] = []
THREADS: set[int] = set()
QUESTION = "Which Vidal do you mean?"
OPTIONS = ["Marc Vidal", "Sophie Vidal"]


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


SOURCE_RULE = {"name": "source", "terms": ["Hello there"], "on": ["output"], "replacement": "[SOURCE]"}


class Excerpts:
    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
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
    def opened(asker) -> Session:
        tools = ToolRegistry([send, Questions(asker).ask_user])
        session = Session(
            config, Agent(config, client, tools=tools, **agent), personas=load_personas(config.personas_dir), asker=asker
        )
        load_memory(session, None)
        return session

    return WebChat(opened)


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
    assert state["context"] == {"used": 0, "limit": 16384, "condensed_from": 12288}
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
    chat.replies.put("allow")
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


def ask_a_question(page, chat, client, **arguments) -> tuple[threading.Thread, dict]:
    client.tool_rounds = [[ToolCall("ask_user", {"question": QUESTION, "options": OPTIONS, **arguments})]]
    thread, result = in_background(page)
    wait_for(lambda: chat.asking)
    return thread, result


def test_a_question_waits_for_the_click_and_the_answer_goes_to_the_model(page, chat, client):
    thread, result = ask_a_question(page, chat, client)
    assert len(client.chats) == 1
    assert page.post("/api/question", json={"answer": "Marc Vidal"}).status_code == 200
    thread.join(5)
    assert client.chats[1][0][-1]["content"] == "The user answered: Marc Vidal"
    assert of_type(result["events"], "question") == [{"type": "question", "question": QUESTION, "options": OPTIONS}]
    # The card shows the question: the call of the tool is not shown as well.
    assert [event["text"] for event in of_type(result["events"], "activity")] == ['[ask_user] answered "Marc Vidal"']


def test_the_answer_can_be_something_the_options_did_not_offer(page, chat, client):
    thread, _ = ask_a_question(page, chat, client)
    page.post("/api/question", json={"answer": "the one in Lyon"})
    thread.join(5)
    assert client.chats[1][0][-1]["content"] == "The user answered: the one in Lyon"


def test_a_question_without_options_has_a_free_answer(page, chat, client):
    thread, result = ask_a_question(page, chat, client, options=None)
    page.post("/api/question", json={"answer": "Lyon"})
    thread.join(5)
    assert of_type(result["events"], "question")[0]["options"] == []


def test_a_question_can_be_left_unanswered(page, chat, client):
    thread, result = ask_a_question(page, chat, client)
    assert page.post("/api/question", json={"answer": None}).status_code == 200
    thread.join(5)
    assert client.chats[1][0][-1]["content"] == NO_ANSWER
    assert of_type(result["events"], "activity")[0]["text"] == "[ask_user] no answer"
    assert of_type(result["events"], "done")


def test_stop_during_a_question_ends_the_answer(page, chat, client):
    thread, result = ask_a_question(page, chat, client)
    page.post("/api/stop")
    thread.join(5)
    assert not thread.is_alive() and of_type(result["events"], "stopped")
    assert len(client.chats) == 1


def test_a_page_closed_during_a_question_does_not_block_the_session(page, chat, client):
    client.tool_rounds = [[ToolCall("ask_user", {"question": QUESTION})]]
    events = chat.start("Who?")
    wait_for(lambda: chat.asking)
    chat.stop()
    while events.get(timeout=5) is not None:
        pass
    assert not chat.asking and ask(page, "Still there?")[-1]["type"] == "state"


def test_an_answer_nobody_asked_for_is_refused(page):
    assert page.post("/api/question", json={"answer": "Marc Vidal"}).status_code == 409
    assert page.post("/api/question", json={}).status_code == 422


def test_a_late_answer_does_not_answer_the_next_question(page, chat, client):
    chat.replies.put("Sophie Vidal")
    thread, _ = ask_a_question(page, chat, client)
    assert len(client.chats) == 1
    page.post("/api/question", json={"answer": "Marc Vidal"})
    thread.join(5)
    assert client.chats[1][0][-1]["content"] == "The user answered: Marc Vidal"


def test_a_click_only_answers_what_is_asked(page, chat, client):
    thread, _ = ask_a_question(page, chat, client)
    assert page.post("/api/approval", json={"choice": "allow"}).status_code == 409
    page.post("/api/question", json={"answer": "Marc Vidal"})
    thread.join(5)
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    thread, _ = in_background(page)
    wait_for(lambda: chat.asking)
    assert page.post("/api/question", json={"answer": "Marc Vidal"}).status_code == 409
    page.post("/api/approval", json={"choice": "deny"})
    thread.join(5)
    assert SENT == []


def test_the_question_is_redacted_before_it_is_shown(config, client):
    rules = [{"name": "source", "terms": ["Claire Benoit"], "on": ["tools"], "action": "redact"}]
    chat = start(config, client, guardrails=Guardrails(parse_rules(rules)))
    client.tool_rounds = [[ToolCall("ask_user", {"question": "Is it Claire Benoit?"})]]
    try:
        thread, result = in_background(page := page_of(chat))
        wait_for(lambda: chat.asking)
        page.post("/api/question", json={"answer": "No"})
        thread.join(5)
    finally:
        chat.close()
    assert "Claire Benoit" not in json.dumps(of_type(result["events"], "question"))


def test_the_question_stays_on_the_page_when_the_feature_comes_back(page, chat, client, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("the question went to the terminal"))
    assert page.post("/api/features/questions", json={"on": False}).status_code == 200
    assert "ask_user" not in chat.session.agent.tools.names()
    assert page.post("/api/features/questions", json={"on": True}).status_code == 200
    thread, result = ask_a_question(page, chat, client)
    page.post("/api/question", json={"answer": "Marc Vidal"})
    thread.join(5)
    assert of_type(result["events"], "question") and client.chats[1][0][-1]["content"] == "The user answered: Marc Vidal"


def web_session(config, client, monkeypatch) -> WebChat:
    """What `run_web` opens, without a model to check."""

    def started(config, asker) -> Session:
        return Session(config, Agent(config, client, tools=ToolRegistry([Questions(asker).ask_user])), asker=asker)

    monkeypatch.setattr("sensai.interfaces.web.server.start_session", started)
    return WebChat(lambda asker: open_session(config, asker))


def test_the_web_command_asks_the_questions_on_the_page(config, client, monkeypatch):
    chat = web_session(config, client, monkeypatch)
    try:
        assert chat.session.asker == chat.ask
    finally:
        chat.close()


def test_the_messages_of_the_page_are_logged(config, client, monkeypatch):
    config.features["logging"] = True
    chat = web_session(config, client, monkeypatch)
    ask(page_of(chat))
    session_id = chat.session.session_id
    chat.close()
    log = InteractionLog(config.log_path)
    logged = log.since(datetime(2000, 1, 1))
    log.close()
    assert [interaction.session_id for interaction in logged] == [session_id]


def test_a_message_is_not_logged_when_the_feature_is_off(config, client, monkeypatch):
    config.features["logging"] = False
    chat = web_session(config, client, monkeypatch)
    try:
        assert chat.session.log is None
    finally:
        chat.close()


def test_the_check_of_an_answer_is_shown():
    started = page_event(CheckStartedEvent(2, "qwen3:1.7b"), [])
    assert started["text"].startswith("Checking 2 sentences with qwen3:1.7b")
    judgement = Judgement([Claim("Marc paid.", "contradicted", "he was paid")], 5, 4)
    checked = page_event(CheckEvent(judgement), [])
    assert checked["warning"] and checked["text"].splitlines()[1] == '  contradicted: "Marc paid." (he was paid)'
    assert not page_event(CheckEvent(Judgement([Claim("Marc was paid.", "supported")], 5, 4)), [])["warning"]
    assert page_event(CheckFailedEvent("no model"), [])["text"] == "The answer could not be checked: no model"


def test_a_message_that_could_not_be_logged_is_reported():
    assert page_event(LogFailedEvent("disk full"), [])["warning"] is True


def test_default_goes_back_to_the_prompt_version_in_use(config, client, tmp_path):
    prompts = PromptStore(tmp_path / "prompts")
    prompts.snapshot("Be clear.")
    prompts.snapshot("Be short.")
    prompts.activate(1)
    config.features["prompt_versions"] = True
    chat = start(config, client, prompts=prompts)
    chat.session.prompts = prompts
    page = page_of(chat)
    try:
        assert page.post("/api/persona", json={"id": "investigator"}).status_code == 200
        assert chat.session.agent.prompt_version is None
        assert page.post("/api/persona", json={"id": None}).status_code == 200
        assert chat.session.agent.memory.system_prompt == "Be clear."
        assert chat.session.agent.prompt_version == 1
    finally:
        chat.close()


def test_everything_runs_on_the_thread_that_opened_the_session(config, client):
    """An SQLite connection refuses any other thread."""
    opened = []
    chat = WebChat(lambda asker: opened.append(threading.get_ident()) or Session(config, Agent(config, client)))
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


def test_the_conversation_can_be_downloaded(page):
    ask(page)
    response = page.get("/api/export/conversation.md")
    assert response.status_code == 200 and "Who approved the transfer?" in response.text
    assert response.headers["content-type"].startswith("text/markdown")
    session = page.get("/api/state").json()["session"]
    assert response.headers["content-disposition"] == f'attachment; filename="sensai-conversation-{session}.md"'


@pytest.mark.parametrize(
    ("suffix", "start"), [("json", b"{"), ("pdf", b"%PDF"), ("html", b"<!doctype html>")],
)
def test_every_format_can_be_downloaded(page, suffix, start):
    ask(page)
    assert page.get(f"/api/export/conversation.{suffix}").content.startswith(start)


def test_the_document_can_be_downloaded(chat, page):
    chat.session.agent.memory.document.apply([Edit("append", "Summary", "Paid in April.")])
    assert "Paid in April." in page.get("/api/export/document.md").text
    assert page.get("/api/state").json()["document"] is True


def test_a_conversation_without_a_message_has_nothing_to_download(page):
    response = page.get("/api/export/conversation.pdf")
    assert response.status_code == 400 and response.json()["detail"] == "No message to export yet."


@pytest.mark.parametrize("path", ["/api/export/conversation.docx", "/api/export/profile.md"])
def test_only_the_known_exports_exist(page, path):
    assert page.get(path).status_code == 422


def test_nothing_is_exported_during_an_answer(chat, page):
    chat.answering.acquire()
    try:
        assert page.get("/api/export/conversation.md").status_code == 409
        assert page.get("/api/shares/conversation").status_code == 409
    finally:
        chat.answering.release()


def test_a_public_link_needs_the_feature(page):
    ask(page)
    assert page.get("/api/state").json()["sharing"] is False
    for response in (page.get("/api/shares/conversation"), page.post("/api/shares/conversation")):
        assert response.status_code == 400 and "The sharing feature is off" in response.json()["detail"]
    assert page.get("/api/shares").json() == {"shares": []}


def test_a_public_link_is_previewed_then_published(chat, config, page):
    config.features["sharing"] = True
    config.public_url = "http://sensai.test"
    chat.session.agent.guardrails = Guardrails(parse_rules([SOURCE_RULE]))
    ask(page)

    preview = page.get("/api/shares/conversation").json()
    assert preview["blocked"] is None
    assert preview["lines"][0] == "Page: Who approved the transfer? (1 message)"
    assert page.get("/api/shares").json() == {"shares": []}

    shown = page.get("/api/shares/conversation/page")
    assert "[SOURCE]" in shown.text and "Hello" not in shown.text
    assert "script-src" not in shown.headers["content-security-policy"]

    link = page.post("/api/shares/conversation").json()["link"]
    share = page.get("/api/shares").json()["shares"][0]
    assert link == share["link"] == f"http://sensai.test/shared/{share['token']}"
    assert share["kind"] == "conversation" and share["title"] == "Who approved the transfer?"
    assert "[SOURCE]" in chat.session.shares.page(share["token"])


def test_a_public_link_can_be_taken_down(chat, config, page):
    config.features["sharing"] = True
    ask(page)
    page.post("/api/shares/conversation")
    token = page.get("/api/shares").json()["shares"][0]["token"]
    assert page.delete(f"/api/shares/{token}").status_code == 200
    assert chat.session.shares.page(token) is None
    assert page.delete(f"/api/shares/{token}").status_code == 404


def test_a_public_link_can_be_taken_down_during_an_answer(chat, config, page):
    config.features["sharing"] = True
    ask(page)
    page.post("/api/shares/conversation")
    token = page.get("/api/shares").json()["shares"][0]["token"]
    chat.answering.acquire()
    try:
        assert page.delete(f"/api/shares/{token}").status_code == 200
    finally:
        chat.answering.release()


def test_a_block_rule_stops_the_public_link(chat, config, page):
    config.features["sharing"] = True
    rule = {"name": "stop", "terms": ["Hello"], "on": ["publish"], "action": "block"}
    chat.session.agent.guardrails = Guardrails(parse_rules([rule]))
    ask(page)
    assert page.get("/api/shares/conversation").json()["blocked"].startswith("Not published: the rule 'stop'")
    response = page.post("/api/shares/conversation")
    assert response.status_code == 400 and chat.session.shares.list() == []


def test_another_site_cannot_publish(config, page):
    config.features["sharing"] = True
    ask(page)
    assert page.post("/api/shares/conversation", headers={"Origin": "http://evil.example"}).status_code == 403

