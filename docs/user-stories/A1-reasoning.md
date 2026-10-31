# A1 Reasoning Loops (ReAct): user stories

## Check how a conclusion was reached before publishing it

As an investigative journalist about to publish an accusation, I want to see each step of Sensai's reasoning and the excerpts it relied on, so that I can check the conclusion before it is printed under my name.

Acceptance criteria:
- `/feature reasoning on` (or `SENSAI_FEATURE_REASONING=true`) turns the reasoning on; it is off by default.
- Each step is shown as soon as it is done (`[thought 1] ...`, then the tool call and its result), before the answer is streamed.
- `/reasoning` lists the questions of the conversation with their number of steps; `/reasoning N` shows the thoughts, the tool calls and the full results of question N.
- The reasoning is never sent back to the model and does not fill its context: only the question and the answer are kept in the history.
- The history stays readable after `/feature reasoning off`, and is cleared by `/new` and `/resume`.

## Follow a lead through the documents

As an investigative journalist working on a financial scheme, I want Sensai to search the documents again with what it has just found, so that it can link a company to its manager, then to the transfers the manager received.

Acceptance criteria:
- With indexed documents and the `rag` feature on, the reasoning can call `search_documents` with its own queries, even when the `tools` feature is off: the search stays on the machine.
- The excerpts it gets name their document, and are given as data, not instructions.
- At each step, the model can only pick a tool that exists, with the arguments of that tool: an invented tool or argument is not possible.
- After `SENSAI_MAX_TOOL_ROUNDS` steps without an answer, or after an invalid step, the reasoning stops with a warning and Sensai answers from what it found.

## Reason without lowering the protection of the source

As an investigative journalist protecting a source, I want the reasoning to follow the same rules as a plain answer, so that no step can send a confidential name out of the machine or show it on screen.

Acceptance criteria:
- A web search chosen during the reasoning waits for my approval, like any critical tool; a refusal and its reason are given to the model, which goes on without the search.
- A tool call whose arguments break a guardrail, like an IBAN in a web search, is blocked and not run.
- The name of the source is redacted in the thoughts shown on screen and kept in `/reasoning`, as in the answer.
