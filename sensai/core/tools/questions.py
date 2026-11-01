from collections.abc import Callable

from sensai.core.tools.registry import ToolError, ToolResult

Asker = Callable[[str, list[str]], str | None]

NO_ANSWER = "The user did not answer. Do not ask again: go on with what you know, or say what is missing."


def _options(options: list[str] | str | None) -> list[str]:
    if options is None:
        return []
    if isinstance(options, str):
        options = [options]
    return list(dict.fromkeys(text for text in (str(option).strip() for option in options) if text))


class Questions:
    def __init__(self, asker: Asker):
        self.asker = asker

    def ask_user(self, question: str, options: list[str] | None = None) -> ToolResult:
        """
        Ask the user a question when the request is ambiguous, or when a piece of information is missing
        that no document and no tool can give: which of two companies with the same name they mean,
        which period to look at, who a nickname refers to. Do not ask what you can find yourself.

        Args:
          question: The question, short and precise
          options: The possible answers when there are only a few; the user can still answer something else

        Returns:
          The user's answer
        """
        question = str(question).strip()
        if not question:
            raise ToolError("ask_user: the question is empty")
        answer = self.asker(question, _options(options))
        if answer is None or not answer.strip():
            return ToolResult(NO_ANSWER, summary="no answer", private=True)
        answer = answer.strip()
        return ToolResult(f"The user answered: {answer}", summary=f'answered "{answer}"', private=True)
