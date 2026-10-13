import hashlib
import re

import pytest

from sensai.core.config import Config


class FakeClient:
    DIMENSIONS = 256

    def __init__(self, answer: str = "Hello there"):
        self.answer = answer
        self.embedded: list[str] = []
        self.chats: list[tuple[list[dict], dict | None]] = []

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

    def chat_stream(self, model: str, messages: list[dict], options: dict | None = None, think: bool | None = None):
        self.chats.append((messages, options))
        self.think = think
        for word in self.answer.split(" "):
            yield word + " "

    def ensure_model(self, model: str) -> None:
        pass


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
    )
