import re
from collections.abc import Generator, Iterator

from sensai.core.agent import Agent
from sensai.core.events import AgentStartedEvent, DoneEvent, Event, ReviewDoneEvent, SourcesEvent, ToolResultEvent
from sensai.core.persona import Persona
from sensai.core.rag.models import SearchResult

REVIEW_PERSONAS = ("investigator", "fact_checker", "editor")
APPROVED = "VERDICT: APPROVED"
REJECTED = "VERDICT: REJECTED"

NO_EVIDENCE = "None: the investigator found no document excerpt and no tool result."

CHECK_PROMPT = f"""Check the answer below, given to the question of a journalist, against the evidence the investigator found.
A claim about the case (a person, an organization, an amount, a date, an event, an accusation) is supported only if the evidence states it: a source the answer cites but that is not in the evidence does not count.
A widely known general fact that the case does not depend on can be marked common knowledge, and does not need a source.
If in doubt, treat it as a claim about the case.
Give a verdict for each claim: supported, common knowledge, unsupported or contradicted, with the reason.
End with a last line that is exactly {APPROVED} if every claim is supported or common knowledge, otherwise {REJECTED}.

Question: {{question}}

Evidence found by the investigator:
{{evidence}}

Answer to check:
{{answer}}"""

FIX_PROMPT = """The fact-checker rejected your answer. Its remarks:
{verdict}

Answer the question again: remove or correct every claim it did not accept, and say which document supports each claim."""

EDIT_PROMPT = """Write a publishable text from the checked answer below, without adding any fact.

Question: {question}

Checked answer:
{answer}"""


class ReviewError(Exception):
    pass


def approved(verdict: str) -> bool:
    """Only an explicit approval on the last line counts: anything else is a rejection."""
    # The markdown can be anywhere on the line, as in "**Verdict:** APPROVED".
    lines = [" ".join(re.sub(r"[*_`]", "", line).split()).strip(".").upper() for line in verdict.splitlines()]
    lines = [line for line in lines if line]
    return bool(lines) and lines[-1] == APPROVED


class Review:
    """The investigator answers, the fact-checker checks until it approves, then the editor writes the text.

    Each agent has its own memory: the fact-checker and the editor only get the text they work on.
    Only the question and the final text go to the history of the main agent.
    """

    def __init__(self, agent: Agent, personas: dict[str, Persona]):
        missing = [persona_id for persona_id in REVIEW_PERSONAS if persona_id not in personas]
        if missing:
            raise ReviewError(f"The review needs the personas {', '.join(missing)}")
        self.agent = agent
        # Only the investigator searches the documents and calls tools.
        self.investigator = self._agent(personas["investigator"], with_tools=True)
        self.fact_checker = self._agent(personas["fact_checker"])
        self.editor = self._agent(personas["editor"])
        # The investigator continues the conversation, so that a follow-up question is understood.
        self.investigator.memory.summary = agent.memory.summary
        self.investigator.memory.turns = list(agent.memory.turns)
        self.sources: dict[int, SearchResult] = {}
        self.tool_results: list[str] = []

    def _agent(self, persona: Persona, with_tools: bool = False) -> Agent:
        main = self.agent
        agent = Agent(
            main.config, main.client,
            retriever=main.retriever if with_tools else None,
            tools=main.tools if with_tools else None,
            guardrails=main.guardrails, persona=persona, approver=main.approver,
        )
        agent.memory.profile = main.memory.profile
        return agent

    def run(self, question: str) -> Iterator[Event]:
        answer = yield from self._ask(self.investigator, question)
        if answer is None:
            return
        max_rounds = self.agent.config.max_review_rounds
        for round_number in range(1, max_rounds + 1):
            check = CHECK_PROMPT.format(question=question, evidence=self._evidence(), answer=answer)
            verdict = yield from self._ask(self.fact_checker, check)
            if verdict is None:
                return
            if approved(verdict):
                text = yield from self._ask(self.editor, EDIT_PROMPT.format(question=question, answer=answer))
                if text is None:
                    return
                yield from self._finish(question, True, text, round_number)
                return
            if round_number < max_rounds:
                answer = yield from self._ask(self.investigator, FIX_PROMPT.format(verdict=verdict))
                if answer is None:
                    return
        yield from self._finish(question, False, answer, max_rounds)

    def _ask(self, agent: Agent, message: str) -> Generator[Event, None, str | None]:
        """Relays the events of the agent and returns its answer, or None when a guardrail blocked the message."""
        yield AgentStartedEvent(agent.persona)
        content = None
        for event in agent.run(message):
            if isinstance(event, DoneEvent):
                content = event.content
            elif isinstance(event, SourcesEvent):
                for source in event.sources:
                    self.sources.setdefault(source.chunk_id, source)
            elif isinstance(event, ToolResultEvent) and not event.result.error:
                self.tool_results.append(f"[{event.call.name}] {event.result.content}")
            yield event
        return content

    def _evidence(self) -> str:
        """The excerpts are numbered in the order they were first found, as in the first answer of the investigator."""
        excerpts = [
            f"[{number}] {source.citation}\n" + "\n\n".join(filter(None, [source.description, source.text]))
            for number, source in enumerate(self.sources.values(), start=1)
        ]
        return "\n\n".join(excerpts + self.tool_results) or NO_EVIDENCE

    def _finish(self, question: str, is_approved: bool, content: str, rounds: int) -> Iterator[Event]:
        self.agent.memory.commit(question, content)
        yield ReviewDoneEvent(is_approved, content, rounds)
