# A6 Semantic Cache: user stories

## Get an answer again without waiting for the model

As an investigative journalist coming back to a case days later, I want Sensai to offer the answer it already gave to a question I ask again in other words, so that I do not wait for the model to write the same answer twice.

Acceptance criteria:
- With the `cache` feature on (the default), every answer given from the documents alone is kept with its question, its sources and its date in `SENSAI_CACHE_PATH`.
- When a new question is at least `SENSAI_CACHE_THRESHOLD` percent close (90 by default) to a question kept in the same context, Sensai shows the question kept, its date and the similarity, then asks `Use this answer? [y]es / [n]o:`.
- `y` shows the answer kept, with its sources, without calling the model; it is added to the conversation like any answer.
- `Cache: N answers kept` is shown at startup, and `/cache` shows how many answers are kept and how often they were reused.

## Never be given the answer to another question

As an investigative journalist about to publish an amount, I want to decide myself whether an answer kept fits my question, so that the answer about money Nordbank sent to Marc Vidal is never given for money Marc Vidal sent to Nordbank.

Acceptance criteria:
- An answer kept is never shown without my confirmation: `n`, `Ctrl+D` or an interface that cannot ask lets the model answer, and an empty or unclear reply asks again.
- Only the questions asked in the same context are compared: the same model, the same conversation so far (persona, profile, earlier turns), the same indexed documents, the same guardrail rules and the same `rag`, `tools` and `reasoning` features. A follow-up question, another persona, a changed document or a new rule gets a new answer, so a name protected by a new rule never shows up in an answer kept before it.
- An answer that called a tool, that a guardrail stopped, that is empty or in JSON is not kept, and a message blocked by a guardrail never reaches the cache.

## Remove the answers I no longer trust

As an investigative journalist who found an error in an answer, I want to remove the answers kept, so that a wrong answer is not offered again.

Acceptance criteria:
- `/cache clear` removes every answer kept and says how many were removed.
- `/feature cache off` stops offering and keeping answers for the session, and keeps those already kept; `/feature cache on` turns it back on.
- When the embedding model is missing or the cache file cannot be opened, Sensai warns once at startup and answers without the cache; an error during a message is a warning, and the model answers.
