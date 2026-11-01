# X1 Web UI: user stories

## Work on a case from a browser page

As an investigative journalist who does not live in a terminal, I want to talk to Sensai in a simple page, so that I can read long sourced answers comfortably and go back to my conversations.

Acceptance criteria:
- `python -m sensai web` starts a server on `http://127.0.0.1:8000` (`--port` to change it), which only this machine can reach.
- Typing a message and pressing Enter shows the answer as it is generated, with the searches, the tool calls and the thoughts above it, and the sources cited below it. Shift+Enter starts a new line.
- The sidebar lists the saved conversations, and clicking one shows it and continues it; `New chat` starts another one and the previous one stays saved.
- The page uses the same conversations as the CLI: one started in the terminal can be continued in the page, and the other way around.
- If Ollama is stopped, the page shows the error and the next message works once it is back.

## See what the agent is doing and steer it

As an investigative journalist, I want to see how full the context is, to choose the persona and the features, and to stop an answer going the wrong way, so that I keep control without typing commands.

Acceptance criteria:
- The composer shows the tokens used out of `SENSAI_NUM_CTX`, updated while the answer is generated, and warns when the oldest turns are about to be condensed.
- The persona list switches the persona for the next message and keeps the history; `Default` goes back to no persona.
- The settings list the features with their description and turn them on or off for this run; a feature that cannot be turned on, like `web_search` without a key, is refused with the reason.
- The stop button ends the answer: what was written stays in the history, and the next message redirects it, as with Ctrl+C in the CLI. Closing or reloading the page stops the answer too.

## Keep the same protections as in the terminal

As an investigative journalist protecting a source, I want the page to apply the same rules as the CLI, so that a nicer interface never means less control over what leaves the machine.

Acceptance criteria:
- A tool that asks for approval stops the answer and shows the exact call with `Allow`, `Always this session` and `Deny`; nothing runs before the click.
- A question of the model stops the answer the same way: nothing is answered for me, and the question never waits in the terminal of the server.
- The guardrails apply as in the chat, and what they redact or block is shown in the conversation.
- The page loads nothing from the internet: it works on an air-gapped machine.
- The answers of the model are never inserted as HTML: a document cannot run a script in the page.
- A web page open in the same browser cannot talk to the server: requests from another origin or for another host name are refused.
