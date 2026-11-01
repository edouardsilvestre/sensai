import pytest
from ollama._utils import convert_function_to_tool

from sensai.core.tools.questions import NO_ANSWER, Questions
from sensai.core.tools.registry import ToolError, ToolRegistry, ToolResult

QUESTION = "Plusieurs sociétés Orion apparaissent. Laquelle ?"


class FakeAsker:
    def __init__(self, answer: str | None):
        self.answer = answer
        self.asked: list[tuple[str, list[str]]] = []

    def __call__(self, question: str, options: list[str]) -> str | None:
        self.asked.append((question, options))
        return self.answer


def test_the_question_reaches_the_user_and_the_answer_the_model():
    asker = FakeAsker("  Orion Delta Consulting ")
    result = Questions(asker).ask_user(f" {QUESTION} ", ["Orion Delta Consulting", "Orion Holding Ltd"])
    assert asker.asked == [(QUESTION, ["Orion Delta Consulting", "Orion Holding Ltd"])]
    assert result.content == "The user answered: Orion Delta Consulting"
    assert result.summary == 'answered "Orion Delta Consulting"'


@pytest.mark.parametrize(
    ("options", "cleaned"),
    [
        (None, []),
        ([], []),
        ([" Lyon ", "", "Chypre", "Lyon", "  "], ["Lyon", "Chypre"]),
        ("Orion Delta Consulting", ["Orion Delta Consulting"]),
    ],
)
def test_the_options_are_cleaned(options, cleaned):
    asker = FakeAsker("Lyon")
    Questions(asker).ask_user(QUESTION, options)
    assert asker.asked[0][1] == cleaned


def test_an_empty_question_is_refused():
    asker = FakeAsker("Lyon")
    with pytest.raises(ToolError, match="the question is empty"):
        Questions(asker).ask_user("   ")
    assert asker.asked == []


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_without_answer_the_model_goes_on_without_asking_again(answer):
    result = Questions(FakeAsker(answer)).ask_user(QUESTION)
    assert result == ToolResult(NO_ANSWER, summary="no answer")
    assert "Do not ask again" in result.content


def test_the_model_sees_what_the_tool_is_for_and_its_arguments():
    tool = Questions(FakeAsker("Lyon")).ask_user
    assert ToolRegistry([tool]).descriptions()["ask_user"].startswith("Ask the user a question when the request is")
    function = convert_function_to_tool(tool).function
    assert function.name == "ask_user"
    assert list(function.parameters.properties) == ["question", "options"]
    assert function.parameters.required == ["question"]
