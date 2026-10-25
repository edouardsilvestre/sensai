from collections.abc import Callable
from dataclasses import dataclass, field

from sensai.core.config import Config
from sensai.core.llm.ollama import OllamaClient
from sensai.core.memory.prompt import build_summary_prompt, summary_word_limit

SUMMARY_HEADER = "Summary of the earlier conversation (older turns were condensed into it):"


@dataclass
class Turn:
    user: str
    assistant: str
    citations: list[str] = field(default_factory=list)

    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.user}, {"role": "assistant", "content": self.assistant}]


Summarizer = Callable[[str, list[Turn]], str]


class ConversationMemory:
    def __init__(self, config: Config, client: OllamaClient | None = None):
        self.config = config
        self.client = client
        self.summary = ""
        self.turns: list[Turn] = []

    @property
    def keep_recent(self) -> int:
        return self.config.keep_recent_turns

    @property
    def limit(self) -> int:
        return self.config.num_ctx * self.config.compact_threshold / 100

    def system_message(self) -> dict | None:
        parts = [self.config.system_prompt] if self.config.system_prompt else []
        if self.summary:
            parts.append(f"{SUMMARY_HEADER}\n{self.summary}")
        return {"role": "system", "content": "\n\n".join(parts)} if parts else None

    def messages(self) -> list[dict]:
        system = self.system_message()
        messages = [system] if system else []
        for turn in self.turns:
            messages += turn.messages()
        return messages

    def commit(self, user: str, assistant: str, citations: list[str] | None = None) -> None:
        self.turns = self.turns + [Turn(user, assistant, list(citations or []))]

    def needs_compaction(self, used_tokens: int) -> bool:
        return bool(self.turns) and used_tokens >= self.limit

    def compact(self, summarize: Summarizer | None = None) -> int:
        summarize = summarize or self.summarize
        # Keep the most recent turns verbatim when possible; past the limit with only recent turns, condense them too.
        # Nothing changes until `summarize` has returned, so a failed or interrupted summary leaves the memory intact.
        split = len(self.turns) - self.keep_recent if len(self.turns) > self.keep_recent else len(self.turns)
        old, recent = self.turns[:split], self.turns[split:]
        summary = summarize(self.summary, old).strip()
        if not summary:
            raise ValueError("The summary is empty")
        self.summary, self.turns = summary, recent
        return len(old)

    def summarize(self, previous_summary: str, turns: list[Turn]) -> str:
        prompt = build_summary_prompt(previous_summary, turns, summary_word_limit(self.config.num_ctx))
        options = {"num_ctx": self.config.num_ctx}
        messages = [{"role": "user", "content": prompt}]
        return self.client.chat(self.config.model, messages, options, think=self.config.think)
