# A3 Human-in-the-Loop: user stories

## Stop a web search that would expose a source

As an investigative journalist protecting a source, I want Sensai to ask me before it sends a web search, so that a query containing a confidential name never leaves the machine without my consent.

Acceptance criteria:
- With web search enabled, when the model calls `web_search`, the CLI prints the exact query (`[web_search] query="..."`), then asks `Allow web_search? The query is sent to an external search engine. [y]es / [n]o / [a]lways this session:` before anything is sent.
- Answering `n` asks for an optional reason, prints `[web_search] refused`, and no request reaches the search engine.
- The model gets the refusal and the reason instead of the results: it can search again without the confidential name, or answer without searching, saying it could not search instead of making up what it would have found.
- An empty or unclear answer asks the question again: nothing is approved by default, and `Ctrl+D` refuses.

## Search freely once I have decided to

As an investigative journalist checking public facts about companies, I want to approve a tool once for the whole session, so that I am not asked again for every search once I have decided that searching is fine.

Acceptance criteria:
- Answering `a` approves the call, and the next calls of the same tool run without a question, with `[web_search] allowed for this session`.
- `/approval` shows the current mode and the tools allowed for the session.
- `/approval critical`, `/approval all` or `/approval off` changes the mode for the session and forgets the tools allowed with `a`; an unknown mode is refused with the list of valid ones.

## Choose how cautious Sensai is

As the technical lead of a newsroom, I want to set when Sensai asks for approval in the configuration, so that a sensitive investigation asks before every tool while an automated job never waits for an answer.

Acceptance criteria:
- `SENSAI_APPROVAL=critical` (the default) asks only before the tools that declare a risk, like `web_search`; `all` asks before every tool call; `off` never asks.
- Any other value stops the startup with an error listing `critical, all, off`.
- A new tool marked with `@requires_approval("reason")` is asked for with that reason, without any other code change.
- When Sensai is used as a library without an approver, critical calls are refused instead of running unattended.
