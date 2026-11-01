# A4 Prompt Versioning: user stories

## Know which system prompt wrote an answer

As the technical lead of a newsroom, I want every change to the system prompt to be kept as a dated version, so that I can tell which prompt produced an answer when a journalist reports a problem.

Acceptance criteria:
- At startup, the system prompt is kept in `SENSAI_PROMPTS_DIR/system.json`; a system prompt changed in configuration becomes a new version, used from then on, and Sensai prints `System prompt changed: kept as version 2. /prompts to list them.`
- `/prompts` lists the versions of the system prompt with their number, date, hash and size, and marks the one in use with `*`.
- `/prompt show N` prints version N, and `/prompt diff N M` the lines that changed between two versions.
- Each saved conversation records the version of its prompt (`prompt_version`).

## Roll back a system prompt that made answers worse

As the technical lead of a newsroom, I want to go back to an earlier version of the system prompt in one command, so that a bad change does not reach journalists while I fix the configuration.

Acceptance criteria:
- `/prompt use N` makes version N the one used by Sensai, at once in the active session and after a restart too.
- Configuring a new system prompt creates a new version upon start, which replaces the rollback.
- An unknown version is refused with the number of versions, and nothing changes.
- `/feature prompt_versions off` goes back to the raw configured system prompt and keeps the versions; an invalid versions file is a warning at startup, and Sensai starts with the configured system prompt.

## Measure whether a new prompt is better

As the technical lead of a newsroom, I want to compare two versions of a system prompt on the same questions, so that I decide on what the answers, their time and their cost show, and not on a feeling.

Acceptance criteria:
- `/prompt compare N M` asks each question of `SENSAI_PROMPT_CASES_PATH` with both versions, alone with the system prompt, at temperature 0 with a fixed seed, and with the same model.
- It prints, for each version, the cases whose answer passes its checks (`must_contain`, `must_not_contain`, ignoring the case), the average time and the average tokens, then each failed check.
- The result is kept with the versions, and `/prompts` shows the last comparison.
- Without a test cases file, the command points to `prompt_cases.example.json`; a model error or Ctrl+C stops the comparison without keeping it.
