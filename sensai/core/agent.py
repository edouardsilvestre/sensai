from collections.abc import Iterator
from typing import Protocol

from sensai.core.approval import NO_APPROVER_REASON, Approval, Approver, approval_reason, refusal
from sensai.core.config import Config
from sensai.core.guardrails.rules import Guardrails
from sensai.core.events import (
    ApprovalEvent,
    CompactionEvent,
    CompactionFailedEvent,
    CompactionStartedEvent,
    DoneEvent,
    Event,
    GuardrailEvent,
    SourcesEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.llm.ollama import LLMError, OllamaClient, ToolCall, Usage
from sensai.core.memory.conversation import ConversationMemory
from sensai.core.persona import Persona
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import build_prompt
from sensai.core.tools.registry import ToolRegistry, ToolResult

# Ends an answer that a block rule stopped, in the history, so that the model knows it was cut.
STOPPED = "[answer stopped by a guardrail]"


class EmptyInputError(ValueError):
    pass


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...


class Agent:
    def __init__(
        self,
        config: Config,
        client: OllamaClient | None = None,
        retriever: Retriever | None = None,
        tools: ToolRegistry | None = None,
        guardrails: Guardrails | None = None,
        persona: Persona | None = None,
        approver: Approver | None = None,
    ):
        self.config = config
        self.client = client or OllamaClient(config.host, config.ollama_api_key)
        self.retriever = retriever
        self.tools = tools or ToolRegistry()
        self.approver = approver
        self.memory = ConversationMemory(config, self.client)
        self.guardrails = guardrails or Guardrails()
        self.persona: Persona | None = None
        if persona is not None:
            self.set_persona(persona)

    def set_persona(self, persona: Persona) -> None:
        self.persona = persona
        self.memory.system_prompt = persona.system_prompt()

    @property
    def messages(self) -> list[dict]:
        return self.memory.messages()

    def _approve(self, call: ToolCall) -> Approval | None:
        reason = approval_reason(self.config.approval, self.tools, call)
        if reason is None:
            return None
        if self.approver is None:
            return Approval(False, NO_APPROVER_REASON)
        return self.approver(call, reason)

    def run(self, user_input: str) -> Iterator[Event]:
        user_input = user_input.strip()
        if not user_input:
            raise EmptyInputError("Empty message")

        guardrails = self.guardrails if self.config.features["guardrails"] else Guardrails()
        checked = guardrails.check_input(user_input)
        for hit in checked.hits:
            yield GuardrailEvent(hit)
        if checked.blocked:
            # Nothing reaches the model, and the message is not kept in the history.
            yield GuardrailEvent(checked.blocked)
            return
        user_input = checked.text

        prompt = user_input
        sources: list[SearchResult] = []
        if self.retriever is not None and self.config.features["rag"]:
            sources = self.retriever.search(user_input, self.config.rag_top_k)
            if sources:
                yield SourcesEvent(sources)
                prompt = build_prompt(user_input, sources)

        pending = self.memory.messages() + [{"role": "user", "content": prompt}]
        usage = Usage()
        options = {"num_ctx": self.config.num_ctx}
        functions = self.tools.tools() if self.config.features["tools"] else []
        # After `max_tool_rounds` calls that requested tools, the next call has to answer.
        max_rounds = self.config.max_tool_rounds
        for round_number in range(max_rounds + 1):
            answer: list[str] = []
            calls: list[ToolCall] = []
            tools = functions if round_number < max_rounds else []
            output = guardrails.output_filter()
            for chunk in self.client.chat_stream(
                self.config.model, pending, options, think=self.config.think, usage=usage,
                tools=tools or None, tool_calls=calls,
            ):
                shown = output.feed(chunk)
                if shown:
                    answer.append(shown)
                    yield TokenEvent(shown)
                if output.blocked:
                    break
            shown = output.close()
            if shown:
                answer.append(shown)
                yield TokenEvent(shown)
            for hit in output.hits.values():
                yield GuardrailEvent(hit)
            if output.blocked:
                yield GuardrailEvent(output.blocked)
                answer.append(f"\n{STOPPED}")
                break
            if not calls:
                break
            # Tool calls and results only live in this turn's messages, like the excerpts.
            pending.append({"role": "assistant", "content": "".join(answer), "tool_calls": [c.message() for c in calls]})
            for call in calls:
                yield ToolCallEvent(call)
                call, checked = guardrails.check_tool_call(call)
                for hit in checked.hits:
                    yield GuardrailEvent(hit)
                if checked.blocked:
                    # The model is told why, so that it can call the tool again without the blocked data.
                    hit = checked.blocked
                    reason = hit.message or f"the arguments contain {hit.found}"
                    result = ToolResult(f"Error: blocked by the guardrail '{hit.rule}' ({hit.found}): {reason}", error=True)
                else:
                    result = self.tools.execute(call)
                approval = self._approve(call)
                if approval is not None:
                    yield ApprovalEvent(call, approval)
                if approval is None or approval.approved:
                    result = self.tools.execute(call)
                else:
                    result = refusal(approval)
                yield ToolResultEvent(call, result)
                pending.append({"role": "tool", "tool_name": call.name, "content": result.content})

        content = "".join(answer)
        self.memory.commit(user_input, content, [source.citation for source in sources], usage)
        yield DoneEvent(content)

        # The turn is saved: from here on, a failure only delays the compaction to the next turn.
        if self.memory.needs_compaction(usage.total_tokens):
            yield CompactionStartedEvent(usage.total_tokens)
            try:
                turns = self.memory.compact()
            except (LLMError, ValueError) as e:
                yield CompactionFailedEvent(str(e))
            else:
                yield CompactionEvent(turns, usage.total_tokens)

