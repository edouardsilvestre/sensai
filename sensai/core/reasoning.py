import json
from dataclasses import dataclass, field

from sensai.core.llm.ollama import ToolCall
from sensai.core.tools.registry import ToolRegistry, ToolResult

ANSWER = "answer"

_INSTRUCTIONS = """\
Work out the answer to the user's message step by step, like an investigator who explains each move.

At each step, reply with:
- thought: what you know so far, what is still missing and what you will do next.
- action: the tool to call next, or "answer" once you know enough to answer.
- arguments: the arguments of the tool, like {{"query": "..."}}; an empty object for "answer".

Rules:
- Call one tool per step, and read its result before the next step.
- Never make up the result of a tool: only rely on what the results say.
- Pick "answer" as soon as you can answer, or when the tools cannot help.
- Write your thoughts in the language of the user's message.

Tools:
{tools}"""

_ANSWER = """\
Before answering the user's message, you reasoned step by step. Your steps and the results of the tools:
{steps}

Now answer the user's message from them, in plain text and in the language of the message.
Only state what the results or the documents say, and say so when they do not answer the message."""


class ReasoningError(Exception):
    pass


@dataclass
class Step:
    thought: str
    action: str
    arguments: dict = field(default_factory=dict)
    result: ToolResult | None = None

    @property
    def final(self) -> bool:
        return self.action == ANSWER

    def call(self) -> ToolCall:
        return ToolCall(self.action, self.arguments)


@dataclass
class Reasoning:
    question: str
    steps: list[Step] = field(default_factory=list)

    def context(self) -> str:
        lines = []
        for number, step in enumerate(self.steps, start=1):
            lines.append(f"Step {number}. Thought: {step.thought}")
            if not step.final:
                lines.append(f"Action: {step.action} {json.dumps(step.arguments, ensure_ascii=False)}")
            if step.result is not None:
                lines.append(f"Result: {step.result.content}")
        return _ANSWER.format(steps="\n".join(lines) or "none")


def step_schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string", "enum": [*actions, ANSWER]},
            "arguments": {"type": "object"},
        },
        "required": ["thought", "action", "arguments"],
    }


def instructions(tools: ToolRegistry) -> str:
    descriptions = tools.descriptions()
    lines = [f"- {name}({', '.join(names)}): {descriptions[name]}" for name, names in tools.parameters().items()]
    return _INSTRUCTIONS.format(tools="\n".join(lines) or 'none, pick "answer".')


def parse_step(text: str, actions: list[str]) -> Step:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ReasoningError(f"The step is not valid JSON: {text[:200]}") from None
    if not isinstance(data, dict):
        raise ReasoningError("The step is not a JSON object")
    thought = data.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        raise ReasoningError("The step has no thought")
    action = data.get("action")
    if action != ANSWER and action not in actions:
        raise ReasoningError(f"Unknown action '{action}'. Available: {', '.join([*actions, ANSWER])}")
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ReasoningError("The arguments of the step are not a JSON object")
    return Step(thought.strip(), action, arguments)
