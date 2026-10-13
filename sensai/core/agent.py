from collections.abc import Iterator
from typing import Protocol

from sensai.core.config import Config
from sensai.core.events import DoneEvent, Event, SourcesEvent, TokenEvent
from sensai.core.llm.ollama import OllamaClient
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import build_prompt


class EmptyInputError(ValueError):
    pass


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...


class Agent:
    def __init__(self, config: Config, client: OllamaClient | None = None, retriever: Retriever | None = None):
        self.config = config
        self.client = client or OllamaClient(config.host)
        self.retriever = retriever
        self.rag_enabled = True
        self.messages: list[dict] = []
        if config.system_prompt:
            self.messages.append({"role": "system", "content": config.system_prompt})

    def run(self, user_input: str) -> Iterator[Event]:
        user_input = user_input.strip()
        if not user_input:
            raise EmptyInputError("Empty message")

        prompt = user_input
        if self.retriever is not None and self.rag_enabled:
            sources = self.retriever.search(user_input, self.config.rag_top_k)
            if sources:
                yield SourcesEvent(sources)
                prompt = build_prompt(user_input, sources)

        pending = self.messages + [{"role": "user", "content": prompt}]
        answer: list[str] = []
        options = {"num_ctx": self.config.num_ctx}
        for chunk in self.client.chat_stream(self.config.model, pending, options, think=self.config.think):
            answer.append(chunk)
            yield TokenEvent(chunk)

        content = "".join(answer)
        self.messages = self.messages + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": content},
        ]
        yield DoneEvent(content)
