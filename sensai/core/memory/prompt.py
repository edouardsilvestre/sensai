from __future__ import annotations

from typing import TYPE_CHECKING

from sensai.core.rag.prompt import cited_numbers

if TYPE_CHECKING:
    from sensai.core.memory.conversation import Turn

_INSTRUCTIONS = """\
Summarize the conversation below in at most {words} words. If a previous summary is given, merge it into the new one.
Keep names, dates, amounts and document names exactly as written, and keep the source of each fact.
Reply with the summary only."""


def summary_word_limit(budget: int) -> int:
    return max(budget // 8 * 0.75, 75)


def _render_turn(turn: Turn) -> str:
    lines = [f"User: {turn.user}", f"Assistant: {turn.assistant}"]
    cited = [number for number in cited_numbers(turn.assistant) if 1 <= number <= len(turn.citations)]
    if cited:
        lines.append("Excerpts cited in this answer: " + "; ".join(f"[{n}] = {turn.citations[n - 1]}" for n in cited))
    return "\n".join(lines)


def build_summary_prompt(previous_summary: str, turns: list[Turn], words: int) -> str:
    parts = [_INSTRUCTIONS.format(words=words)]
    if previous_summary:
        parts.append(f"Previous summary:\n{previous_summary}")
    parts.append("Conversation:\n" + "\n\n".join(_render_turn(turn) for turn in turns))
    return "\n\n".join(parts)
