import pytest

from sensai.core.config import Config
from sensai.core.memory.conversation import SUMMARY_HEADER, ConversationMemory, Turn


def make(system_prompt: str = "", keep_recent: int = 4, num_ctx: int = 8192, threshold: int = 75, client=None):
    config = Config(
        system_prompt=system_prompt, num_ctx=num_ctx, compact_threshold=threshold, keep_recent_turns=keep_recent
    )
    return ConversationMemory(config, client)


def test_messages_start_with_the_system_prompt():
    memory = make("Be precise.")
    memory.commit("Hi", "Hello")
    assert memory.messages() == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_no_system_message_without_prompt_or_summary():
    memory = make()
    memory.commit("Hi", "Hello")
    assert memory.messages()[0]["role"] == "user"


def test_summary_follows_the_system_prompt():
    memory = make("Be precise.")
    memory.summary = "Alice approved the transfer."
    assert memory.messages() == [
        {"role": "system", "content": f"Be precise.\n\n{SUMMARY_HEADER}\nAlice approved the transfer."}
    ]


def fill(memory: ConversationMemory, count: int) -> None:
    for number in range(count):
        memory.commit(f"question {number}", f"answer {number}")


def test_limit_is_the_threshold_share_of_the_budget():
    assert make(num_ctx=8192, threshold=75).limit == 6144


def test_compaction_is_needed_only_past_the_threshold():
    memory = make(num_ctx=8192, threshold=75, keep_recent=2)
    fill(memory, 3)
    assert not memory.needs_compaction(6143)
    assert memory.needs_compaction(6144)


def test_recent_turns_alone_are_never_compacted():
    memory = make(num_ctx=8192, threshold=75, keep_recent=4)
    fill(memory, 4)
    assert not memory.needs_compaction(8192)


def test_compaction_condenses_old_turns_and_keeps_recent_ones():
    memory = make("Be precise.", keep_recent=2)
    fill(memory, 5)
    calls = []

    def summarize(previous: str, turns: list[Turn]) -> str:
        calls.append((previous, [turn.user for turn in turns]))
        return "  Alice approved the transfer.\n"

    assert memory.compact(summarize) == 3
    assert calls == [("", ["question 0", "question 1", "question 2"])]
    assert [turn.user for turn in memory.turns] == ["question 3", "question 4"]
    assert memory.messages()[0] == {
        "role": "system",
        "content": f"Be precise.\n\n{SUMMARY_HEADER}\nAlice approved the transfer.",
    }


def test_summary_is_rolled_into_the_next_compaction():
    memory = make(keep_recent=1)
    fill(memory, 3)
    memory.compact(lambda previous, turns: "first summary")
    fill(memory, 2)
    seen = []
    memory.compact(lambda previous, turns: seen.append(previous) or "second summary")
    assert seen == ["first summary"]
    assert memory.summary == "second summary"


def test_nothing_to_compact_with_only_recent_turns():
    memory = make(keep_recent=4)
    fill(memory, 2)
    assert memory.compact(lambda previous, turns: "unused") == 0
    assert memory.summary == ""


@pytest.mark.parametrize("failure", [RuntimeError("model crashed"), KeyboardInterrupt()])
def test_failed_summary_leaves_memory_unchanged(failure):
    memory = make(keep_recent=1)
    fill(memory, 3)
    turns = list(memory.turns)

    def summarize(previous, turns):
        raise failure

    with pytest.raises(type(failure)):
        memory.compact(summarize)
    assert memory.turns == turns
    assert memory.summary == ""


def test_empty_summary_is_rejected():
    memory = make(keep_recent=1)
    fill(memory, 3)
    with pytest.raises(ValueError):
        memory.compact(lambda previous, turns: "   ")
    assert len(memory.turns) == 3


def test_the_model_writes_the_summary_by_default(config, client):
    config.keep_recent_turns = 1
    memory = ConversationMemory(config, client)
    fill(memory, 3)

    assert memory.compact() == 2
    assert memory.summary == client.summary
    prompt = client.summaries[0][0]["content"]
    assert "User: question 0" in prompt and "User: question 1" in prompt
    assert "question 2" not in prompt
