# M4 Artifact & State Document: user stories

## Build the investigation file while I talk

As an investigative journalist questioning a leak for hours, I want Sensai to keep a file of what has been established (people, companies, dates, open questions) apart from the chat, so that I have a clean deliverable at the end instead of a transcript I must read again.

Acceptance criteria:
- After each answer, the document is updated without any command, and the CLI prints what changed: `[document] added to Timeline, added to People and companies. /document to read it.`
- `/document` shows the document in Markdown, with its sections (`Summary`, `People and companies`, `Timeline`, `Open questions` by default); an empty document lists the sections it will fill.
- The document is not the chat: questions, greetings and the answers themselves are not copied into it, and a message that establishes nothing new leaves it unchanged, without any output.
- `/document export enquete-alvor.md` writes it to a Markdown file, and the document is saved with the conversation: `--resume` and `/resume` bring it back, `/new` starts an empty one.

## Correct a fact without losing the rest

As an investigative journalist, I want a correction made in the conversation ("Marc Vidal is the CEO, not the CFO") to change the right passage of the document and nothing else, so that the file stays accurate without being rewritten or corrupted every turn.

Acceptance criteria:
- The model sends edits as JSON (`append`, `replace`, `rewrite`, `add_section`), each touching one section; Ollama's structured outputs force a reply that can always be parsed.
- A `replace` changes only the quoted passage, which must be found exactly once in that section; otherwise the edit is rejected and the document is unchanged: `[warning] document edit rejected (replace in 'People and companies'): the text to find is not in the section`.
- An edit naming an unknown section, a line the section already has, or more than 8 edits in one turn are rejected with their reason, while the valid edits of the same turn are applied.
- An invalid reply or an Ollama error during the update keeps the answer and the document as they were, with a warning: `The document was not updated: ...`.
- `/document undo` puts back the document as it was before the last answer, and the undo is saved.

## Never publish what must stay hidden

As an investigative journalist protecting a source, I want the document to follow the same guardrails as the answers, so that the file I share with my editor can never name the source or carry personal data.

Acceptance criteria:
- With a `source` rule redacting `Claire Benoit`, an edit that names her is written with `[SOURCE]` instead, and the CLI reports the redaction.
- An edit containing data that a rule blocks (an IBAN with a `block` rule) is not written: `document edit rejected (append in 'Timeline'): blocked by the guardrail iban (IBAN)`.
- A message blocked by the guardrails, or an answer stopped by them, does not update the document.
- With `SENSAI_FEATURE_DOCUMENT=false` (or `/feature document off`), no call is made to update the document and it is not sent to the model; it is kept and `/document` still shows it.
