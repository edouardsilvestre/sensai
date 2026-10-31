from collections.abc import Callable
from dataclasses import dataclass, field, replace

from sensai.core.config import Config
from sensai.core.document.document import Document
from sensai.core.llm.ollama import OllamaClient, Usage
from sensai.core.memory.profile import PROFILE_HEADER
from sensai.core.memory.prompt import build_summary_prompt, summary_word_limit

SUMMARY_HEADER = "Summary of the earlier conversation (older turns were condensed into it):"
DOCUMENT_HEADER = (
    "Working document, kept up to date separately after each answer: refer to it, but do not rewrite it in your answers."
)


@dataclass
class Turn:
    user: str
    assistant: str
    citations: list[str] = field(default_factory=list)
    # The working document before this turn: a branch that replaces the turn starts from it.
    document: str = field(default="", compare=False)

    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.user}, {"role": "assistant", "content": self.assistant}]


@dataclass
class Branch:
    """One version of the conversation, with its own history and document: what happens in one never reaches another."""

    number: int = 1
    parent: int | None = None
    # In the parent, the number of the turn this branch replaces, and the message that replaced it.
    turn: int = 0
    label: str = ""
    summary: str = ""
    turns: list[Turn] = field(default_factory=list)
    document: Document = field(default_factory=Document)


Summarizer = Callable[[str, list[Turn]], str]


class ConversationMemory:
    def __init__(self, config: Config, client: OllamaClient | None = None):
        self.config = config
        self.client = client
        self.system_prompt = config.system_prompt
        self.summary = ""
        # The user's profile, sent after the system prompt while the profile feature is on.
        self.profile = ""
        # The deliverable of the conversation (M4): never condensed, it only changes through edits.
        self.document = Document()
        self.last_usage: Usage | None = None
        self.summary_usage: Usage | None = None
        self.turns: list[Turn] = []
        # The active branch only names itself: its content is in the attributes above.
        self.branch = Branch()
        self.other_branches: list[Branch] = []

    @property
    def keep_recent(self) -> int:
        return self.config.keep_recent_turns

    @property
    def limit(self) -> int:
        return self.config.num_ctx * self.config.compact_threshold / 100

    def _system_message(self) -> dict | None:
        parts = [self.system_prompt] if self.system_prompt else []
        if self.profile and self.config.features["profile"]:
            parts.append(f"{PROFILE_HEADER}\n{self.profile}")
        if self.config.features["document"] and not self.document.is_empty:
            parts.append(f"{DOCUMENT_HEADER}\n{self.document.markdown().strip()}")
        if self.summary:
            parts.append(f"{SUMMARY_HEADER}\n{self.summary}")
        return {"role": "system", "content": "\n\n".join(parts)} if parts else None

    def messages(self) -> list[dict]:
        system = self._system_message()
        messages = [system] if system else []
        for turn in self.turns:
            messages += turn.messages()
        return messages

    def commit(
        self, user: str, assistant: str, citations: list[str] | None = None, usage: Usage | None = None
    ) -> None:
        self.turns = self.turns + [Turn(user, assistant, list(citations or []), self.document.markdown())]
        self.last_usage = usage

    def clear(self) -> None:
        self.summary, self.turns, self.last_usage = "", [], None
        self.document = Document()
        self.branch, self.other_branches = Branch(), []

    def _active(self) -> Branch:
        return replace(self.branch, summary=self.summary, turns=self.turns, document=self.document)

    def _activate(self, branch: Branch) -> None:
        self.branch, self.summary, self.turns, self.document = branch, branch.summary, branch.turns, branch.document
        self.last_usage = None

    def branches(self) -> list[Branch]:
        return sorted([*self.other_branches, self._active()], key=lambda branch: branch.number)

    def fork(self, number: int, label: str) -> None:
        """Starts a branch just before the turn `number` (from 1), which the active branch keeps, like all its turns."""
        turn = self.turns[number - 1]
        # Sessions saved before the branches do not have the document of each turn.
        document = Document(turn.document) if turn.document else Document()
        branch = Branch(
            max(branch.number for branch in self.branches()) + 1, self.branch.number, number, label,
            self.summary, self.turns[: number - 1], document,
        )
        self.other_branches.append(self._active())
        self._activate(branch)

    def switch(self, number: int) -> bool:
        target = next((branch for branch in self.other_branches if branch.number == number), None)
        if target is None:
            return False
        self.other_branches = [branch for branch in self.other_branches if branch is not target] + [self._active()]
        self._activate(target)
        return True

    def discard_branch(self) -> None:
        """Back to the parent of the active branch, which is forgotten: its first message never got an answer."""
        parent = next(branch for branch in self.other_branches if branch.number == self.branch.parent)
        self.other_branches.remove(parent)
        self._activate(parent)

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
        self.summary_usage = Usage()
        return self.client.chat(self.config.model, messages, options, think=self.config.think, usage=self.summary_usage)
