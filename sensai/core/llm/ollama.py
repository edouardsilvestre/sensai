"""Thin wrapper around the Ollama client.

This is the only module that imports `ollama`: the rest of the code talks to
`OllamaClient` and only sees the exceptions defined here.
"""

from collections.abc import Iterator

import httpx
import ollama


class LLMError(Exception):
    """Base class for every model-related error."""


class LLMConnectionError(LLMError):
    """The Ollama server cannot be reached."""


class ModelNotFoundError(LLMError):
    """The requested model is not available on the Ollama server."""

    def __init__(self, model: str, available: list[str]):
        self.model = model
        self.available = available
        hint = ", ".join(available) if available else "none (use `ollama pull <model>`)"
        super().__init__(f"Model '{model}' is not available. Installed models: {hint}")


class OllamaClient:
    def __init__(self, host: str):
        self.host = host
        self._client = ollama.Client(host=host)

    def list_models(self) -> list[str]:
        try:
            return [m.model for m in self._client.list().models if m.model]
        except (ConnectionError, httpx.HTTPError):
            raise LLMConnectionError(f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?") from None
        except ollama.ResponseError as e:
            raise LLMError(f"Ollama error: {e.error}") from None

    def chat_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        """Yield the assistant's answer chunk by chunk."""
        try:
            for chunk in self._client.chat(model=model, messages=messages, stream=True):
                if chunk.message.content:
                    yield chunk.message.content
        except (ConnectionError, httpx.HTTPError):
            raise LLMConnectionError(f"Lost connection to Ollama at {self.host}.") from None
