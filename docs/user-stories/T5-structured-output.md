# T5 Structured output: user stories

## Get an answer a program can read

As an investigative journalist, I want an answer in JSON that always has the same form, so that I can load the people, companies and events of the case into a spreadsheet, a database or a graph tool without copying them by hand.

Acceptance criteria:
- `/json QUESTION` answers from the indexed documents, with the profile and the guardrails, like a plain message.
- The model is forced by a schema sent to Ollama (`format`) to answer `{"answer", "claims", "entities", "events"}`: each claim with the numbers of its excerpts, each entity with a type (`person`, `company`, `place` or `other`), each event with its date.
- The JSON is printed once complete and indented, followed by the excerpts it cites.
- `/json` without a question prints its usage and sends nothing to the model.

## Never trust a reply because it looks like JSON

As the technical lead of a newsroom, I want every structured reply checked by the code against its schema, so that a cut or malformed reply never reaches a tool that reads it.

Acceptance criteria:
- A reply that is not valid JSON, misses a field or has a wrong type is refused with what is wrong (`$.claims[0].sources is missing`), and the raw reply is shown.
- A claim citing an excerpt that was not retrieved is flagged, as in a prose answer.
- An answer with a schema is given no tool: the schema would stop the model from calling it.

## Let the code decide, not the model's wording

As an investigative journalist, I want the approval of a review and the edits of the working document computed from structured replies, so that a fact-checker's phrasing or a stray sentence can never approve an answer or corrupt the document.

Acceptance criteria:
- The fact-checker answers `{"claims": [{"claim", "reason", "verdict"}]}`; the answer is approved only when it lists at least one claim and every verdict is `supported` or `common knowledge`.
- The reason of each verdict comes before the verdict, so that the model reasons before deciding.
- The document editor answers `{"edits": [...]}`; an edit with a missing or wrong field is rejected with the reason, the others are applied.
