import hashlib
import json
from collections.abc import Generator, Iterator

from sensai.core.approval import NO_APPROVER_REASON, Approval, Approver, approval_reason, refusal
from sensai.core.cache import CacheError, CachedAnswer, ConfirmCache, SemanticCache
from sensai.core.config import Config
from sensai.core.document.document import DocumentError, Edit, Rejected
from sensai.core.document.editor import DocumentEditor, model_editor
from sensai.core.evaluation.judge import check, evidence, sentences
from sensai.core.examples import ExampleLibrary, examples_message
from sensai.core.guardrails.rules import Guardrails
from sensai.core.events import (
    ApprovalEvent,
    CheckEvent,
    CheckFailedEvent,
    CheckStartedEvent,
    CacheFailedEvent,
    CacheHitEvent,
    CompactionEvent,
    CompactionFailedEvent,
    CompactionStartedEvent,
    DocumentEvent,
    DocumentFailedEvent,
    DoneEvent,
    Event,
    ExamplesEvent,
    ExamplesFailedEvent,
    GuardrailEvent,
    ReasoningStoppedEvent,
    SourcesEvent,
    ThoughtEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.llm.ollama import LLMError, OllamaClient, ToolCall, Usage
from sensai.core.memory.conversation import ConversationMemory
from sensai.core.persona import Persona
from sensai.core.prompts import PromptStore
from sensai.core.rag.models import SearchResult
from sensai.core.rag.prompt import build_prompt
from sensai.core.reasoning import Reasoning, ReasoningError, instructions, parse_step, step_schema, tool_messages
from sensai.core.tools.documents import DocumentSearch, Retriever
from sensai.core.tools.registry import ToolRegistry, ToolResult

# Ends an answer that a block rule stopped, in the history, so that the model knows it was cut.
STOPPED = "[answer stopped by a guardrail]"
# Ends an answer the user interrupted, in the history, so that the model knows it was cut.
INTERRUPTED = "[answer interrupted by the user]"


def cut_answer(text: str) -> str:
    """What the history keeps of an answer the user interrupted."""
    return "\n".join(filter(None, [text, INTERRUPTED]))


class EmptyInputError(ValueError):
    pass


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
        editor: DocumentEditor | None = None,
        cache: SemanticCache | None = None,
        confirm_cache: ConfirmCache | None = None,
        examples: ExampleLibrary | None = None,
        prompts: PromptStore | None = None,
    ):
        self.config = config
        self.client = client or OllamaClient(config.host, config.ollama_api_key)
        self.retriever = retriever
        self.tools = tools or ToolRegistry()
        self.approver = approver
        self.memory = ConversationMemory(config, self.client)
        self.editor = editor or model_editor(self.client, config)
        self.guardrails = guardrails or Guardrails()
        self.examples = examples
        self.prompts = prompts
        self.prompt_version: int | None = None
        self.reasoning: list[Reasoning] = []
        self.cache = cache
        self.confirm_cache = confirm_cache
        self.keeps_document = True
        # The turn being answered and the text shown so far, for `interrupt`.
        self.answering: tuple[str, list[SearchResult]] | None = None
        self.shown: list[str] = []
        # The last question sent to the model, as the guardrails let it through.
        self.asked = ""
        self.persona: Persona | None = None
        if self.prompts is not None and self.config.features["prompt_versions"]:
            version = self.prompts.active()
            self.prompt_version = version.number if version else None
            if version is not None:
                self.memory.system_prompt = version.prompt
        if persona is not None:
            self.set_persona(persona)

    def set_persona(self, persona: Persona) -> None:
        self.persona = persona
        self.prompt_version = None
        self.memory.system_prompt = persona.system_prompt()

    @property
    def messages(self) -> list[dict]:
        return self.memory.messages()

    @property
    def active_guardrails(self) -> Guardrails:
        return self.guardrails if self.config.features["guardrails"] else Guardrails()

    def interrupt(self) -> bool:
        """
        Call it when `run` was stopped midway. The question and the text already shown go to the history, so that the
        next message can redirect the answer. The document does not change: a cut answer establishes nothing.
        """
        if self.answering is None:
            return False
        (user_input, sources), self.answering = self.answering, None
        text = cut_answer("".join(self.shown).lstrip())
        self.memory.commit(user_input, text, [source.citation for source in sources])
        return True

    def _approve(self, call: ToolCall, tools: ToolRegistry) -> Approval | None:
        reason = approval_reason(self.config.approval, tools, call)
        if reason is None:
            return None
        if self.approver is None:
            return Approval(False, NO_APPROVER_REASON)
        return self.approver(call, reason)

    def _run_tool(
        self, call: ToolCall, guardrails: Guardrails, tools: ToolRegistry
    ) -> Generator[Event, None, tuple[ToolCall, ToolResult]]:
        call, checked = guardrails.check_tool_call(call)
        for hit in checked.hits:
            yield GuardrailEvent(hit)
        yield ToolCallEvent(call)
        if checked.blocked:
            # The model is told why, so that it can call the tool again without the blocked data.
            hit = checked.blocked
            reason = hit.message or f"the arguments contain {hit.found}"
            result = ToolResult(f"Error: blocked by the guardrail '{hit.rule}' ({hit.found}): {reason}", error=True)
        else:
            # Asked after the guardrails: the user approves the redacted call, the one that runs.
            approval = self._approve(call, tools)
            if approval is not None:
                yield ApprovalEvent(call, approval)
            result = tools.execute(call) if approval is None or approval.approved else refusal(approval)
        self.shown.append(self._kept_result(call, result, guardrails))
        yield ToolResultEvent(call, result)
        return call, result

    def _reason(
        self, question: str, pending: list[dict], options: dict, guardrails: Guardrails, sources: list[SearchResult]
    ) -> Generator[Event, None, Reasoning]:
        reasoning = Reasoning(question)
        self.reasoning.append(reasoning)
        tools = self.tools if self.config.features["tools"] else ToolRegistry()
        if self.retriever is not None and self.config.features["rag"]:
            tools = tools.extended([DocumentSearch(self.retriever, self.config.rag_top_k, sources).search_documents])
        actions = tools.names()
        schema = step_schema(tools)
        messages = pending[:-1] + [{"role": "system", "content": instructions(tools)}] + pending[-1:]
        native = self.client.can_call_tools(self.config.model)
        for number in range(1, self.config.max_tool_rounds + 1):
            reply = self.client.chat(self.config.model, messages, options, think=self.config.think, format=schema)
            try:
                step = parse_step(reply, actions)
            except ReasoningError as e:
                yield ReasoningStoppedEvent(str(e))
                return reasoning
            output = guardrails.output_filter()
            step.thought = output.feed(step.thought) + output.close()
            for hit in output.hits.values():
                yield GuardrailEvent(hit)
            if output.blocked:
                yield GuardrailEvent(output.blocked)
                yield ReasoningStoppedEvent("a thought was stopped by a guardrail")
                return reasoning
            reasoning.steps.append(step)
            yield ThoughtEvent(number, step.thought)
            if step.final:
                return reasoning
            call, step.result = yield from self._run_tool(step.call(), guardrails, tools)
            step.arguments = call.arguments
            messages += tool_messages(step, native)
        yield ReasoningStoppedEvent(f"no answer after {self.config.max_tool_rounds} steps")
        return reasoning

    def _context(self) -> str:
        fingerprint = getattr(self.retriever, "fingerprint", None)
        examples = (
            self.examples is not None and self.examples.examples
            and self.config.examples_k > 0 and self.config.features["examples"]
        )
        context = {
            "model": self.config.model,
            "messages": self.memory.messages(),
            "documents": fingerprint() if fingerprint and self.config.features["rag"] else None,
            "guardrails": self.active_guardrails.fingerprint(),
            "examples": [self.examples.fingerprint(), self.config.examples_k] if examples else None,
            "features": {name: self.config.features[name] for name in ("rag", "tools", "reasoning")},
        }
        return hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _find_in_cache(self, question: str) -> tuple[list[float], str, CachedAnswer | None]:
        context = self._context()
        vector = self.cache.embed(question)
        return vector, context, self.cache.find(vector, context)

    def update_document(self, user_input: str, answer: str, guardrails: Guardrails) -> Iterator[Event]:
        """Asks for the edits of this turn and applies them. A failure leaves the document as it was."""
        try:
            edits = self.editor(self.memory.document, user_input, answer)
        except (LLMError, DocumentError) as e:
            yield DocumentFailedEvent(str(e))
            return
        # The document is a deliverable: what may not be shown in an answer may not be written in it either.
        allowed: list[Edit] = []
        blocked: list[Rejected] = []
        for edit in edits:
            # The section name is written too: add_section makes it a heading of the document.
            name, text = guardrails.check("output", edit.section), guardrails.check("output", edit.text)
            for hit in name.hits + text.hits:
                yield GuardrailEvent(hit)
            hit = name.blocked or text.blocked
            if hit:
                blocked.append(Rejected(edit, f"blocked by the guardrail {hit.rule} ({hit.found})"))
            else:
                edit.section, edit.text = name.text, text.text
                allowed.append(edit)
        report = self.memory.document.apply(allowed)
        report.rejected = blocked + report.rejected
        if report:
            yield DocumentEvent(report)

    def _kept_result(self, call: ToolCall, result: ToolResult, guardrails: Guardrails) -> str:
        """
        A tool result as an interrupted turn keeps it, for the model to go on from it: unlike an answer,
        this one reaches the history and the saved session, so it goes through the same rules as an answer.
        """
        output = guardrails.output_filter()
        content = output.feed(result.content) + output.close()
        return f"\n[{call.name}] {'result withheld by a guardrail' if output.blocked else content}\n"

    def _check(self, question: str, answer: str, given: str) -> Iterator[Event]:
        yield CheckStartedEvent(len(sentences(answer)), self.config.judge_model or self.config.model)
        try:
            judgement = check(self.client, self.config, question, answer, given)
        except LLMError as e:
            yield CheckFailedEvent(str(e))
            return
        yield CheckEvent(judgement)

    def run(self, user_input: str, format: dict | None = None) -> Iterator[Event]:
        try:
            yield from self._answer(user_input, format)
        except Exception:
            # Unlike an interruption, a failure leaves nothing to keep.
            self.answering, self.shown = None, []
            raise

    def _answer(self, user_input: str, format: dict | None) -> Iterator[Event]:
        self.answering, self.shown = None, []
        user_input = user_input.strip()
        if not user_input:
            raise EmptyInputError("Empty message")

        guardrails = self.active_guardrails
        checked = guardrails.check_input(user_input)
        for hit in checked.hits:
            yield GuardrailEvent(hit)
        if checked.blocked:
            # Nothing reaches the model, and the message is not kept in the history.
            yield GuardrailEvent(checked.blocked)
            return
        user_input = checked.text

        vector, context = None, ""
        cache_on = self.cache is not None and self.confirm_cache is not None and self.config.features["cache"]
        if cache_on and format is None:
            try:
                vector, context, cached = self._find_in_cache(user_input)
            except (LLMError, CacheError) as e:
                yield CacheFailedEvent(str(e))
                cached = None
            if cached is not None and self.confirm_cache(cached):
                yield CacheHitEvent(cached)
                if cached.sources:
                    yield SourcesEvent(cached.sources)
                yield TokenEvent(cached.answer)
                self.memory.commit(user_input, cached.answer, [source.citation for source in cached.sources])
                yield DoneEvent(cached.answer)
                return

        prompt = user_input
        sources: list[SearchResult] = []
        if self.retriever is not None and self.config.features["rag"]:
            sources = self.retriever.search(user_input, self.config.rag_top_k)
            if sources:
                yield SourcesEvent(sources)
                prompt = build_prompt(user_input, sources)

        # A JSON answer cut halfway is of no use to anyone: only a prose answer is kept.
        # A copy: the reasoning adds the excerpts it finds, and an interrupted turn cites them too.
        sources = list(sources)
        self.asked = user_input
        self.answering = None if format else (user_input, sources)
        pending = self.memory.messages() + [{"role": "user", "content": prompt}]
        if self.examples is not None and self.config.features["examples"] and format is None:
            try:
                chosen = self.examples.closest(user_input, self.config.examples_k)
            except LLMError as e:
                yield ExamplesFailedEvent(str(e))
                chosen = []
            if chosen:
                yield ExamplesEvent(chosen)
                pending.insert(-1, {"role": "system", "content": examples_message(chosen)})
        usage = Usage()
        options = {"num_ctx": self.config.num_ctx}
        # The schema constrains every token of the answer: the model could not call a tool.
        functions = self.tools.tools() if self.config.features["tools"] and format is None else []
        used_tools = False
        reasoning = None
        if self.config.features["reasoning"] and format is None:
            found = len(sources)
            reasoning = yield from self._reason(user_input, pending, options, guardrails, sources)
            if len(sources) > found:
                yield SourcesEvent(list(sources))
            pending.insert(-1, {"role": "system", "content": reasoning.context()})
            functions = []
            used_tools = any(step.result is not None for step in reasoning.steps)
        # After `max_tool_rounds` calls that requested tools, the next call has to answer.
        max_rounds = self.config.max_tool_rounds
        for round_number in range(max_rounds + 1):
            answer: list[str] = []
            calls: list[ToolCall] = []
            tools = functions if round_number < max_rounds else []
            output = guardrails.output_filter()
            for chunk in self.client.chat_stream(
                self.config.model, pending, options, think=self.config.think, usage=usage,
                tools=tools or None, tool_calls=calls, format=format,
            ):
                shown = output.feed(chunk)
                if shown:
                    answer.append(shown)
                    self.shown.append(shown)
                    yield TokenEvent(shown)
                if output.blocked:
                    break
            shown = output.close()
            if shown:
                answer.append(shown)
                self.shown.append(shown)
                yield TokenEvent(shown)
            for hit in output.hits.values():
                yield GuardrailEvent(hit)
            if output.blocked:
                yield GuardrailEvent(output.blocked)
                answer.append(f"\n{STOPPED}")
                break
            if not calls:
                break
            used_tools = True
            # Tool calls and results only live in this turn's messages, like the excerpts.
            pending.append({"role": "assistant", "content": "".join(answer), "tool_calls": [c.message() for c in calls]})
            for call in calls:
                call, result = yield from self._run_tool(call, guardrails, self.tools)
                pending.append({"role": "tool", "tool_name": call.name, "content": result.content})

        content = "".join(answer)
        self.answering = None
        self.memory.commit(user_input, content, [source.citation for source in sources], usage)
        yield DoneEvent(content)
        if vector is not None and not used_tools and not output.blocked and content.strip():
            try:
                self.cache.add(user_input, vector, context, content, sources)
            except CacheError as e:
                yield CacheFailedEvent(str(e))
        if self.keeps_document and self.config.features["document"] and content.strip() and not output.blocked:
            yield from self.update_document(user_input, content, guardrails)
        checks = self.keeps_document and self.config.features["fact_check"] and format is None
        if checks and sentences(content) and not output.blocked:
            results = [
                message["content"] for message in pending
                if message.get("role") == "tool" and not message["content"].startswith("Error:")
            ]
            if reasoning is not None:
                results += [
                    step.result.content for step in reasoning.steps
                    if step.result is not None and not step.result.error and step.action != "search_documents"
                ]
            yield from self._check(user_input, content, evidence(sources, results))

        # The turn is saved: from here on, a failure only delays the compaction to the next turn.
        if self.memory.needs_compaction(usage.total_tokens):
            yield CompactionStartedEvent(usage.total_tokens)
            try:
                turns = self.memory.compact()
            except (LLMError, ValueError) as e:
                yield CompactionFailedEvent(str(e))
            else:
                yield CompactionEvent(turns, usage.total_tokens)

