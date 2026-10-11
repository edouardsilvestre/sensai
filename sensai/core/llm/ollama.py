"""Thin wrapper around the Ollama client.

This is the only module that imports `ollama`: the rest of the code talks to
`OllamaClient` and only sees the exceptions defined here.
"""

from collections.abc import Iterator

import httpx
import ollama


class LLMError(Exception):
    """Base class for every model-related error."""


class OllamaClient:
    def __init__(self, host: str):
        self.host = host
        self._client = ollama.Client(host=host)
