# A5 Persona: user stories

## Have the investigation checked without explaining it again

As an investigative journalist, I want to switch Sensai from the investigator to the fact-checker in the middle of a session, so that the claims of my draft are challenged against what the investigation established, without having to explain the whole case again.

Acceptance criteria:
- `python -m sensai -p investigator` (or `SENSAI_PERSONA=investigator`) starts as the investigator: the startup message names it and the prompt shows `[investigator] >`.
- `/persona fact_checker` confirms the switch and the prompt becomes `[fact_checker] >`.
- After the switch, the fact-checker can use the facts given before it (the turns and the summary of older turns are kept).
- The fact-checker answers with a verdict per claim (supported, unsupported or contradicted) instead of looking for new leads.
- `/persona nope` is refused with the list of available personas, and the active persona does not change.

## Turn the findings into an article

As an investigative journalist, I want to hand the established findings to an editor persona, so that I get a publishable draft that adds no fact the investigation did not establish.

Acceptance criteria:
- `python -m sensai -p editor` starts the session as the editor, whatever `SENSAI_PERSONA` says.
- After `/persona editor` in a running session, the draft is written from the facts of the conversation and keeps who said or wrote what.
- `/personas` lists the investigator, the fact-checker and the editor with their role, and marks the active one with `*`.

## Adapt Sensai to a newsroom without touching the code

As the technical lead of a newsroom, I want to add a profile (a legal reviewer, a data journalist...) by writing a single JSON file, so that each team gets an assistant suited to its job without changing Sensai's code.

Acceptance criteria:
- A `legal.json` file with `name`, `role`, `tone` and `scope` in `personas/` (or in `SENSAI_PERSONAS_DIR`) appears in `/personas` at the next start.
- `SENSAI_PERSONA=legal` in `.env` starts the sessions with that persona; without `SENSAI_PERSONA`, Sensai starts without persona, uses `SENSAI_SYSTEM_PROMPT` and suggests `/persona ID` to pick one.
- A file with a missing field or invalid JSON stops the startup with an error naming the file and the problem.
