from sensai.core.memory.conversation import Turn
from sensai.core.memory.prompt import build_summary_prompt, summary_word_limit


def test_cited_excerpts_are_resolved_to_their_source():
    turn = Turn("Who approved it?", "Alice approved it [2].", citations=["leak/a.pdf, p. 1", "leak/mail.eml, p. 2"])
    prompt = build_summary_prompt("", [turn], words=300)

    assert "Excerpts cited in this answer: [2] = leak/mail.eml, p. 2" in prompt
    assert "leak/a.pdf" not in prompt


def test_unknown_excerpt_numbers_are_not_resolved():
    turn = Turn("Q", "Claim [7].", citations=["leak/a.pdf"])
    assert "Excerpts cited" not in build_summary_prompt("", [turn], words=300)


def test_previous_summary_conversation_and_size_are_included():
    prompt = build_summary_prompt("Bob owns Acme.", [Turn("Hi", "Hello")], words=300)

    assert "Previous summary:\nBob owns Acme." in prompt
    assert "Conversation:\nUser: Hi\nAssistant: Hello" in prompt
    assert "at most 300 words" in prompt


def test_first_summary_has_no_previous_one():
    assert "Previous summary" not in build_summary_prompt("", [Turn("Hi", "Hello")], words=300)


def test_summary_size_follows_the_context_window():
    assert summary_word_limit(8192) == 768
    assert summary_word_limit(512) == 75
