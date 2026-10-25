from ollama import ChatResponse, Message

from sensai.core.llm.ollama import OllamaClient, Usage


class FakeOllama:
    def chat(self, stream: bool, **kwargs):
        if not stream:
            return ChatResponse(message=Message(role="assistant", content="A summary."), done=True)
        return self._stream()

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
