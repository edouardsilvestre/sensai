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
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.llm.ollama import LLMError, OllamaClient, ToolCall, Usage
from sensai.core.memory.conversation import ConversationMemory
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import build_prompt
from sensai.core.tools.registry import ToolRegistry


class EmptyInputError(ValueError):
    pass


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...


class Agent:
    def __init__(
        self,
        config: Config,
        client: OllamaClient | None = None,
        retriever: Retriever | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.config = config
        self.client = client or OllamaClient(config.host, config.ollama_api_key)
        self.retriever = retriever
        self.rag_enabled = True
        self.tools = tools or ToolRegistry()
        self.tools_enabled = True
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
        usage = Usage()
        options = {"num_ctx": self.config.num_ctx}
        functions = self.tools.tools() if self.tools_enabled else []
        # After `max_tool_rounds` calls that requested tools, the next call has to answer.
        max_rounds = self.config.max_tool_rounds
        for round_number in range(max_rounds + 1):
            answer: list[str] = []
            calls: list[ToolCall] = []
            tools = functions if round_number < max_rounds else []
            for chunk in self.client.chat_stream(
                self.config.model, pending, options, think=self.config.think, usage=usage,
                tools=tools or None, tool_calls=calls,
            ):
                answer.append(chunk)
                yield TokenEvent(chunk)
            if not calls:
                break
            # Tool calls and results only live in this turn's messages, like the excerpts.
            pending.append({"role": "assistant", "content": "".join(answer), "tool_calls": [c.message() for c in calls]})
            for call in calls:
                yield ToolCallEvent(call)
                result = self.tools.execute(call)
                yield ToolResultEvent(call, result)
                pending.append({"role": "tool", "tool_name": call.name, "content": result.content})

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

