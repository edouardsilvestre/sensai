from sensai.core.memory.conversation import SUMMARY_HEADER, ConversationMemory


def test_messages_start_with_the_system_prompt():
    memory = ConversationMemory("Be precise.")
    memory.commit("Hi", "Hello")
    assert memory.messages() == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_no_system_message_without_prompt_or_summary():
    memory = ConversationMemory()
    memory.commit("Hi", "Hello")
    assert memory.messages()[0]["role"] == "user"


def test_summary_follows_the_system_prompt():
    memory = ConversationMemory("Be precise.")
    memory.summary = "Alice approved the transfer."
    assert memory.messages() == [
        {"role": "system", "content": f"Be precise.\n\n{SUMMARY_HEADER}\nAlice approved the transfer."}
    ]

