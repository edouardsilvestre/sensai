import httpx
import ollama
import pytest
from ollama import ChatResponse, Message, ShowResponse, WebSearchResponse

from sensai.core.llm.ollama import LLMError, OllamaClient, ToolCall, Usage, WebResult


class FakeOllama:
    capabilities = ["completion", "tools"]
    search_error: Exception | None = None

    def web_search(self, query: str, max_results: int = 3):
        if self.search_error:
            raise self.search_error
        return WebSearchResponse(results=[{"title": "Alvor", "url": "https://alvor.example/", "content": "Lisbon"}, {}])

    def show(self, model: str):
        return ShowResponse(model_info={}, capabilities=self.capabilities)

    def chat(self, stream: bool, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("tools"):
            return self._tool_stream()
        if not stream:
            return ChatResponse(
                message=Message(role="assistant", content="A summary."), done=True, prompt_eval_count=300, eval_count=20
            )
        return self._stream()

    def _tool_stream(self):
        call = Message.ToolCall(function=Message.ToolCall.Function(name="web_search", arguments={"query": "Alvor"}))
        yield ChatResponse(message=Message(role="assistant", content="", tool_calls=[call]), done=False)
        yield ChatResponse(message=Message(role="assistant", content=""), done=True, prompt_eval_count=5, eval_count=1)

    def _stream(self):
        yield ChatResponse(message=Message(role="assistant", content="Hello "), done=False)
        yield ChatResponse(message=Message(role="assistant", content="there"), done=False)
        yield ChatResponse(
            message=Message(role="assistant", content=""), done=True, prompt_eval_count=42, eval_count=7
        )


def make_client() -> OllamaClient:
    client = OllamaClient("http://localhost:11434")
    client._client = FakeOllama()
    return client


def test_usage_is_filled_from_the_final_chunk():
    usage = Usage()
    chunks = list(make_client().chat_stream("model", [], usage=usage))

    assert chunks == ["Hello ", "there"]
    assert usage == Usage(prompt_tokens=42, completion_tokens=7)
    assert usage.total_tokens == 49


def test_usage_is_optional():
    assert list(make_client().chat_stream("model", [])) == ["Hello ", "there"]


def test_chat_returns_the_whole_answer():
    assert make_client().chat("model", []) == "A summary."


def test_chat_sends_the_json_schema_as_format():
    client = make_client()
    schema = {"type": "object", "properties": {"edits": {"type": "array"}}}
    client.chat("model", [], format=schema)
    assert client._client.kwargs["format"] == schema


def test_chat_fills_usage():
    usage = Usage()
    make_client().chat("model", [], usage=usage)
    assert usage == Usage(prompt_tokens=300, completion_tokens=20)


def test_chat_replies_freely_without_a_schema():
    client = make_client()
    client.chat("model", [])
    assert client._client.kwargs["format"] is None


def test_tool_calls_are_collected():
    client = make_client()
    def web_search(query: str) -> str:
        return query

    calls: list[ToolCall] = []
    assert list(client.chat_stream("model", [], tools=[web_search], tool_calls=calls)) == []
    assert calls == [ToolCall("web_search", {"query": "Alvor"})]
    assert client._client.kwargs["tools"] == [web_search]


def test_tool_call_message_format():
    assert ToolCall("web_search", {"query": "Alvor"}).message() == {
        "function": {"name": "web_search", "arguments": {"query": "Alvor"}}
    }


def test_can_call_tools_reads_the_model_capabilities():
    client = make_client()
    assert client.can_call_tools("model")
    client._client.capabilities = ["completion"]
    assert not client.can_call_tools("model")


def test_web_search_returns_the_results():
    assert make_client().web_search("Alvor", 2) == [
        WebResult("Alvor", "https://alvor.example/", "Lisbon"), WebResult("", "", ""),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ValueError("Authorization header with Bearer token is required"), "OLLAMA_API_KEY"),
        (ollama.ResponseError("unauthorized", 401), "unauthorized"),
        (httpx.ConnectError("no route"), "ollama.com"),
    ],
)
def test_web_search_errors(error, expected):
    client = make_client()
    client._client.search_error = error
    with pytest.raises(LLMError, match=expected):
        client.web_search("Alvor", 2)


def test_api_key_is_sent_as_a_bearer_token():
    client = OllamaClient("http://localhost:11434", api_key="secret")
    assert client._client._client.headers["authorization"] == "Bearer secret"
