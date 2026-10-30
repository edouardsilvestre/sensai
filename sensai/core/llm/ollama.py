from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx
import ollama


class LLMError(Exception):
    pass


class LLMConnectionError(LLMError):
    pass


class ModelNotFoundError(LLMError):
    def __init__(self, model: str, available: list[str] | None):
        self.model = model
        self.available = available
        if available is None:
            hint = "unknown (run `ollama list`)"
        elif available:
            hint = ", ".join(available)
        else:
            hint = f"none (run `ollama pull {model}`)"
        super().__init__(f"Model '{model}' is not available. Installed models: {hint}")


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCall:
    name: str
    arguments: dict

    def message(self) -> dict:
        return {"function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class WebResult:
    title: str
    url: str
    content: str


class OllamaClient:
    def __init__(self, host: str, api_key: str = ""):
        self.host = host
        # The key is only needed by the web search, which goes to ollama.com instead of `host`.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = ollama.Client(host=host, headers=headers)

    def list_models(self) -> list[str]:
        try:
            return [m.model for m in self._client.list().models if m.model]
        except (ConnectionError, httpx.HTTPError) as e:
            raise LLMConnectionError(f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?") from e
        except ollama.ResponseError as e:
            raise LLMError(f"Ollama error: {e.error}") from e

    def ensure_model(self, model: str) -> None:
        available = self.list_models()
        normalized = {name.removesuffix(":latest") for name in available}
        if model.removesuffix(":latest") not in normalized:
            raise ModelNotFoundError(model, available)

    def can_call_tools(self, model: str) -> bool:
        try:
            capabilities = self._client.show(model).capabilities
        except (ConnectionError, httpx.HTTPError) as e:
            raise LLMConnectionError(f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?") from e
        except ollama.ResponseError as e:
            raise self._response_error(model, e) from e
        # Older Ollama servers do not report capabilities: let the model try.
        return capabilities is None or "tools" in capabilities

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        think: bool | None = None,
        usage: Usage | None = None,
        tools: list[Callable | dict] | None = None,
        tool_calls: list[ToolCall] | None = None,
        format: dict | None = None,
    ) -> Iterator[str]:
        try:
            for chunk in self._client.chat(
                model=model, messages=messages, stream=True, options=options, think=think, tools=tools, format=format
            ):
                if chunk.message.content:
                    yield chunk.message.content
                if chunk.message.tool_calls and tool_calls is not None:
                    tool_calls += [
                        ToolCall(call.function.name, dict(call.function.arguments)) for call in chunk.message.tool_calls
                    ]
                if chunk.done and usage is not None:
                    usage.prompt_tokens = chunk.prompt_eval_count or 0
                    usage.completion_tokens = chunk.eval_count or 0
        except (ConnectionError, httpx.HTTPError) as e:
            raise LLMConnectionError(f"Lost connection to Ollama at {self.host}.") from e
        except ollama.ResponseError as e:
            raise self._response_error(model, e) from e

    def chat(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        think: bool | None = None,
        usage: Usage | None = None,
        format: dict | None = None,
    ) -> str:
        """`format` is a JSON schema the answer must follow (Ollama's structured outputs)."""
        try:
            response = self._client.chat(
                model=model, messages=messages, stream=False, options=options, think=think, format=format
            )
        except (ConnectionError, httpx.HTTPError) as e:
            raise LLMConnectionError(f"Lost connection to Ollama at {self.host}.") from e
        except ollama.ResponseError as e:
            raise self._response_error(model, e) from e
        if usage is not None:
            usage.prompt_tokens = response.prompt_eval_count or 0
            usage.completion_tokens = response.eval_count or 0
        return response.message.content or ""

    def embed(self, model: str, texts: list[str], options: dict | None = None) -> list[list[float]]:
        try:
            response = self._client.embed(model=model, input=texts, options=options)
        except (ConnectionError, httpx.HTTPError) as e:
            raise LLMConnectionError(f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?") from e
        except ollama.ResponseError as e:
            raise self._response_error(model, e) from e
        if len(response.embeddings) != len(texts):
            raise LLMError(f"Ollama returned {len(response.embeddings)} embeddings for {len(texts)} texts")
        return [list(vector) for vector in response.embeddings]

    def web_search(self, query: str, max_results: int) -> list[WebResult]:
        try:
            response = self._client.web_search(query, max_results=max_results)
        except ValueError as e:
            raise LLMError("The web search needs an Ollama API key: set OLLAMA_API_KEY") from e
        except (ConnectionError, httpx.HTTPError) as e:
            raise LLMConnectionError("Cannot reach ollama.com for the web search.") from e
        except ollama.ResponseError as e:
            raise LLMError(f"Ollama web search error: {e.error}") from e
        return [WebResult(r.title or "", r.url or "", r.content or "") for r in response.results]

    def _response_error(self, model: str, error: ollama.ResponseError) -> LLMError:
        if error.status_code == 404:
            try:
                available = self.list_models()
            except LLMError:
                available = None
            return ModelNotFoundError(model, available)
        return LLMError(f"Ollama error: {error.error}")
