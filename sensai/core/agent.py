"""The agent loop: keeps the conversation history and streams answers."""

from collections.abc import Iterator

from sensai.core.config import Config
from sensai.core.events import DoneEvent, Event, TokenEvent
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

    def run(self, user_input: str) -> Iterator[Event]:
        """Send `user_input` with the full history and stream the answer.

        The turn is committed to the history only once the answer is complete,
        so an error or an interruption leaves the history unchanged.
        """
        user_input = user_input.strip()
        if not user_input:
            raise EmptyInputError("Empty message")

        pending = self.messages + [{"role": "user", "content": user_input}]
        answer: list[str] = []
        for chunk in self.client.chat_stream(self.config.model, pending):
            answer.append(chunk)
            yield TokenEvent(chunk)

        content = "".join(answer)
        self.messages = pending + [{"role": "assistant", "content": content}]
        yield DoneEvent(content)
