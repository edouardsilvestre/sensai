# T3 Web Search: user stories

## Check a company found in the leak against public sources

As an investigative journalist, I want Sensai to look up a company named in the leak on the web, so that I can tell whether it has a real public footprint (website, registry entry, press coverage) or looks like a shell company.

Acceptance criteria:
- With `SENSAI_FEATURE_WEB_SEARCH=true`, asking "Alvor Consulting a-t-elle une existence publique ?" makes the model call `web_search`, and the CLI prints `[web_search] query="..."` before the query is sent.
- The CLI then prints the number of results and their domains (`[web_search] 5 results: ...`).
- The answer cites the URLs of the results it relies on, and says plainly when the search found nothing relevant.
- When the search fails (no network, invalid `OLLAMA_API_KEY`, quota reached), a warning explains why and the model still answers, saying the search failed.

## Keep control over what leaves the machine

As an investigative journalist protecting a source, I want web search to be off unless I turn it on, and to see every query sent, so that nothing about the investigation reaches a search engine without my knowledge.

Acceptance criteria:
- Without `SENSAI_FEATURE_WEB_SEARCH=true`, no tool is sent to the model and no network request is made apart from the local Ollama server; `/tools` says that no tool is enabled.
- With web search enabled, `/feature web_search off` stops the model from searching for the rest of the session, and `/feature web_search on` allows it again (it refuses without `OLLAMA_API_KEY`).
- Each query is printed before it is sent, never after.
- The search results are only sent for the current message: the conversation history only keeps a short record of the search.

## Get recent facts the model was not trained on

As a journalist preparing a publication, I want to ask about recent public events, so that my article does not rely on outdated knowledge from the model's training data.

Acceptance criteria:
- A question about a recent event ("Quelle est la dernière version stable de Python ?") triggers a search instead of an answer from memory.
- The model may search several times in one turn (refining its query), up to `SENSAI_MAX_TOOL_ROUNDS` rounds (5 by default), then it has to answer.
- With a model that cannot call tools (such as `llama3`), Sensai warns at startup and keeps working without web search instead of failing.
- Instructions written inside a web page ("ignore your instructions...") are treated as data: the results are marked as untrusted content.
