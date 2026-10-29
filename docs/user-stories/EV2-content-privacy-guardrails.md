# EV2 Content & Privacy Guardrails: user stories

## Never let the identity of my source appear

As an investigative journalist protecting a source, I want Sensai to remove the name and the codename of my source from every answer and every web search, so that a screenshot, a copied paragraph or a search query can never expose who gave me the documents.

Acceptance criteria:
- With a `source` rule listing `"Claire Benoit"` and `"PHOENIX"` in `.sensai/guardrails.json` (`SENSAI_GUARDRAILS_PATH`), an answer that would name her shows `[SOURCE]` instead, as it is streamed: the name never appears on screen, even when it arrives split across two chunks of the answer.
- The name is found whatever its case, accents or spaces (`claire  BENOÎT`), but not inside another word.
- The redacted answer is the one kept in the history and saved with the session, and the CLI reports it: `[guardrail] source: 1 passage redacted in the answer (SOURCE)`.
- `/guardrails` lists the rule as `2 protected terms` without printing the protected names.

## Keep personal data on the machine

As an investigative journalist working on leaked bank statements and emails, I want personal data (emails, phone numbers, IBANs, card numbers, social security numbers, IP addresses) to be stopped before it is sent to a web search, so that the people named in the leak are not exposed to a search engine.

Acceptance criteria:
- Without any rules file, a `web_search` call whose query contains an IBAN is not sent: the model gets `blocked by the guardrail 'personal-data' (IBAN)` and searches again without it, and the CLI shows the warning.
- The same IBAN in `run_python` code, which runs without network, is not blocked.
- Checksums are verified (IBAN mod 97, card Luhn, social security key), so that amounts, dates and invoice numbers are not taken for personal data.
- A rule can instead `redact` or only `warn` (for example to flag bank details before they are quoted in an article), in the answer as well as in the tool calls.

## Refuse requests that cross the line

As the editor responsible for an investigation, I want Sensai to refuse requests that target the private life of a person, so that the tool is used to investigate facts and not to find where someone lives or to track them.

Acceptance criteria:
- With a `judge` rule describing what is not allowed ("finding the home address, family or daily movements of a private person..."), each message is first checked by the local model; "Où habite Marc Vidal ?" is blocked with the reason given by the model, and nothing is sent to answer it.
- A blocked message is not kept in the history and not saved; a plain question about the documents is answered as usual.
- An unclear verdict from the model blocks the message (fail closed), and so does an Ollama error during the check.
- Simpler rules can block with a pattern (`"pattern": "adresse personnelle"`) without any model call; an invalid rules file stops Sensai at startup with the reason, instead of running without its rules.
