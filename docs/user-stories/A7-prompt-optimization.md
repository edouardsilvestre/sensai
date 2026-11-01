# A7 Automated Prompt Optimization: user stories

## Teach the model how I want it to answer

As an investigative journalist working with a small local model, I want to keep the answers I find good as examples, so that the model answers the next questions the same way: citing its excerpts and saying what the documents do not say, instead of making it up.

Acceptance criteria:
- `/example add` keeps the last answer of the conversation with its question and the excerpts it cites, in `SENSAI_EXAMPLES_PATH`, and prints `Answer kept as example N`.
- For each new question, the `SENSAI_EXAMPLES_K` examples whose question is the closest (2 by default), above a similarity of 0.5, are sent to the model just before the question, with the instruction to answer the same way without reusing their facts.
- The examples never go to the history of the conversation, and the step-by-step reasoning gets them too.
- No examples are shipped with Sensai: without `/example add`, the prompt does not change and nothing more is computed.

## See what guided an answer

As an investigative journalist checking an answer before using it, I want to know which examples the model was shown, so that I can tell whether an example shaped the answer.

Acceptance criteria:
- Before the answer, `[examples] 2 examples: "...", "..."` names the questions of the examples shown.
- `/examples` lists the examples kept with their date, and marks with `*` those shown for the last answer.
- When the question cannot be embedded, a warning says that no examples were shown, and the model answers without them.

## Keep control of what the model imitates

As an investigative journalist, I want to decide alone which answers are examples, so that a wrong or badly written answer is never imitated.

Acceptance criteria:
- Only plain answers can be kept: an answer of `/json` or `/review`, or an answer stopped by a guardrail, is refused, and so is an answer already kept.
- `/example remove N` removes example N; `/feature examples off` stops showing the examples, and keeps them.
- The examples stay in a file on the machine, which can be edited by hand; an invalid file is a warning at startup, and Sensai starts without the examples.
