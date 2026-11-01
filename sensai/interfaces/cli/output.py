import sys
import textwrap

SHORT_TEXT = 70


def error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr)


def warning(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} s"


def shorten(text: str) -> str:
    return textwrap.shorten(text, width=SHORT_TEXT, placeholder="...")


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def score(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}/5"
