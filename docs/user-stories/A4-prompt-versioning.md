# A4 Prompt Versioning: user stories

## Know which prompt wrote an answer

As the technical lead of a newsroom, I want every change to a persona prompt to be kept as a dated version, so that I can tell which prompt produced an answer when a journalist reports a problem.

Acceptance criteria:
- At startup, the prompt of each persona file is kept in `SENSAI_PROMPTS_DIR/<id>.json`; a persona file edited since the last start becomes a new version, used from then on, and Sensai prints `Prompt of editor changed: kept as version 2`.
- `/prompts [PERSONA]` lists the versions of the active persona, or of the persona named, with their number, date, hash and size, and marks the one in use with `*`.
- `/prompt show N` prints version N, and `/prompt diff N M` the lines that changed between two versions.
- Each saved conversation records its persona and the version of its prompt (`persona` and `prompt_version`).

## Roll back a prompt that made the answers worse

As the technical lead of a newsroom, I want to go back to an earlier version of a persona prompt in one command, so that a bad change does not reach the journalists while I fix the persona file.

Acceptance criteria:
- `/prompt use N` makes version N the one used by the persona, at once for the active persona and for the agents of `/review`, and after a restart too.
- Editing the persona file again creates a new version, which replaces the rollback.
- An unknown version or persona is refused with the number of versions or the list of personas, and nothing changes.
- `/feature prompt_versions off` goes back to the prompts of the persona files and keeps the versions; an invalid versions file is a warning at startup, and Sensai starts with the persona files.

## Measure whether a new prompt is better

As the technical lead of a newsroom, I want to compare two versions of a prompt on the same questions, so that I decide on what the answers, their time and their cost show, and not on a feeling.

Acceptance criteria:
- `/prompt compare N M` asks each question of `SENSAI_PROMPT_CASES_PATH` with both versions, alone with the system prompt, at temperature 0 with a fixed seed, and with the same model.
- It prints, for each version, the cases whose answer passes its checks (`must_contain`, `must_not_contain`, ignoring the case), the average time and the average tokens, then each failed check.
- The result is kept with the versions, and `/prompts` shows the last comparison.
- Without a test cases file, the command points to `prompt_cases.example.json`; a model error or Ctrl+C stops the comparison without keeping it.
