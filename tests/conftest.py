import hashlib
import re

import pytest

from sensai.core.config import Config


class FakeClient:
    DIMENSIONS = 256

    def __init__(self, answer: str = "Hello there", summary: str = "Facts: Alice approved the transfer."):
        self.answer = answer
        self.summary = summary
        self.usage = (100, 10)
        self.embedded: list[str] = []
        self.chats: list[tuple[list[dict], dict | None]] = []
        self.summaries: list[list[dict]] = []
        # The reply to the document editor, which asks for JSON edits, and the prompts it got.
        self.edits = '{"edits": []}'
        self.edit_requests: list[list[dict]] = []
        # Tool calls returned by the next calls made with tools, one list per call.
        self.tool_rounds: list[list] = []
        self.tools_seen: list[list[dict] | None] = []
        self.steps: list[str] = []
        self.steps_seen: list[tuple[list[dict], dict]] = []
        self.formats_seen: list[dict | None] = []
        self.capable = True

    def embed(self, model: str, texts: list[str], options: dict | None = None) -> list[list[float]]:
        self.embedded += texts
        self.embed_options = options
        return [self.vector(text) for text in texts]

    @classmethod
    def vector(cls, text: str) -> list[float]:
        vector = [0.0] * cls.DIMENSIONS
        vector[0] = 0.01
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % cls.DIMENSIONS] += 1.0
        return vector

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        think: bool | None = None,
        usage=None,
        tools: list | None = None,
        tool_calls: list | None = None,
        format: dict | None = None,
    ):
        self.chats.append((list(messages), options))
        self.formats_seen.append(format)
        self.think = think
        self.tools_seen.append(tools)
        if tools and self.tool_rounds:
            tool_calls += self.tool_rounds.pop(0)
        else:
            for word in self.answer.split(" "):
                yield word + " "
        if usage is not None:
            usage.prompt_tokens, usage.completion_tokens = self.usage

    def chat(
        self, model: str, messages: list[dict], options: dict | None = None, think: bool | None = None, usage=None,
        format: dict | None = None,
    ) -> str:
        if format is not None and "anyOf" in format:
            self.steps_seen.append((list(messages), format))
            return self.steps.pop(0) if self.steps else '{"thought": "I can answer.", "action": "answer"}'
        if format is not None:
            self.edit_requests.append(messages)
            return self.edits
        self.summaries.append(messages)
        if usage is not None:
            usage.prompt_tokens, usage.completion_tokens = 300, 20
        return self.summary

    def ensure_model(self, model: str) -> None:
        pass

    def can_call_tools(self, model: str) -> bool:
        return self.capable


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        model="chat-model",
        embed_model="fake-embed",
        index_path=tmp_path / "index" / "index.db",
        rag_top_k=3,
        chunk_size=200,
        chunk_overlap=40,
        sessions_dir=tmp_path / "sessions",
        profile_path=tmp_path / "profile.md",
        memory_path=tmp_path / "memory.db",
        mcp_path=tmp_path / "mcp.json",
        cache_path=tmp_path / "cache.db",
        examples_path=tmp_path / "examples.json",
    )
