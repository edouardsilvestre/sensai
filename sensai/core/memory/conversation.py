from dataclasses import dataclass

SUMMARY_HEADER = "Summary of the earlier conversation (older turns were condensed into it):"


@dataclass
class Turn:
    user: str
    assistant: str

    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.user}, {"role": "assistant", "content": self.assistant}]


class ConversationMemory:
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.summary = ""
        self.turns: list[Turn] = []

    def system_message(self) -> dict | None:
        parts = [self.system_prompt] if self.system_prompt else []
        if self.summary:
            parts.append(f"{SUMMARY_HEADER}\n{self.summary}")
        return {"role": "system", "content": "\n\n".join(parts)} if parts else None

    def messages(self) -> list[dict]:
        system = self.system_message()
        messages = [system] if system else []
        for turn in self.turns:
            messages += turn.messages()
        return messages

    def commit(self, user: str, assistant: str) -> None:
        self.turns = self.turns + [Turn(user, assistant)]
