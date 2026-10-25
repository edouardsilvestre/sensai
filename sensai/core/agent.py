from collections.abc import Iterator
from typing import Protocol

from sensai.core.config import Config
from sensai.core.events import (
    CompactionEvent,
    CompactionFailedEvent,
    CompactionStartedEvent,
    DoneEvent,
    Event,
    SourcesEvent,
    TokenEvent,
)
from sensai.core.llm.ollama import LLMError, OllamaClient, Usage
from sensai.core.memory.conversation import ConversationMemory
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
        self.memory = ConversationMemory(config, self.client)

    @property
    def messages(self) -> list[dict]:
        return self.memory.messages()

    def run(self, user_input: str) -> Iterator[Event]:
        user_input = user_input.strip()
        if not user_input:
            raise EmptyInputError("Empty message")

        prompt = user_input
        sources: list[SearchResult] = []
        if self.retriever is not None and self.rag_enabled:
            sources = self.retriever.search(user_input, self.config.rag_top_k)
            if sources:
                yield SourcesEvent(sources)
                prompt = build_prompt(user_input, sources)

        pending = self.memory.messages() + [{"role": "user", "content": prompt}]
        answer: list[str] = []
        usage = Usage()
        options = {"num_ctx": self.config.num_ctx}
        for chunk in self.client.chat_stream(self.config.model, pending, options, think=self.config.think, usage=usage):
            answer.append(chunk)
            yield TokenEvent(chunk)

        content = "".join(answer)
        self.memory.commit(user_input, content, [source.citation for source in sources], usage)
        yield DoneEvent(content)

        # The turn is saved: from here on, a failure only delays the compaction to the next turn.
        if self.memory.needs_compaction(usage.total_tokens):
            yield CompactionStartedEvent(usage.total_tokens)
            try:
                turns = self.memory.compact()
            except (LLMError, ValueError) as e:
                yield CompactionFailedEvent(str(e))
            else:
                yield CompactionEvent(turns, usage.total_tokens)

