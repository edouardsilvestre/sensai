"""Events emitted by the agent. Interfaces decide how to render them."""

from dataclasses import dataclass


@dataclass
class TokenEvent:
    """A chunk of the assistant's answer."""

    text: str


@dataclass
class DoneEvent:
    """The assistant's answer is complete."""

    content: str


Event = TokenEvent | DoneEvent
