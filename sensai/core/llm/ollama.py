from collections.abc import Iterator
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


class OllamaClient:
    def __init__(self, host: str):
        self.host = host
        self._client = ollama.Client(host=host)

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

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        think: bool | None = None,
        usage: Usage | None = None,
    ) -> Iterator[str]:
        try:
            for chunk in self._client.chat(model=model, messages=messages, stream=True, options=options, think=think):
                if chunk.message.content:
                    yield chunk.message.content
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
    ) -> str:
        try:
            response = self._client.chat(model=model, messages=messages, stream=False, options=options, think=think)
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

    def _response_error(self, model: str, error: ollama.ResponseError) -> LLMError:
        if error.status_code == 404:
            try:
                available = self.list_models()
            except LLMError:
                available = None
            return ModelNotFoundError(model, available)
        return LLMError(f"Ollama error: {error.error}")
