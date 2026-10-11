"""The agent loop: keeps the conversation history and streams answers."""

from sensai.core.config import Config
from sensai.core.llm.ollama import OllamaClient


class EmptyInputError(ValueError):
    """Raised when the user sends an empty message."""


class Agent:
    def __init__(self, config: Config, client: OllamaClient | None = None):
        self.config = config
        self.client = client or OllamaClient(config.host)
        self.messages: list[dict] = []
        if config.system_prompt:
            self.messages.append({"role": "system", "content": config.system_prompt})

