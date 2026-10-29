# M1 Session Persistence & Profiles: user stories

## Pick up an investigation where I left it

As an investigative journalist working on a leak for weeks, I want to close Sensai in the evening and continue the same conversation the next morning, so that I do not have to explain the case again or lose the leads found the day before.

Acceptance criteria:
- Every answered message saves the conversation in `.sensai/sessions/` (`SENSAI_SESSIONS_DIR`), without any command; an interrupted answer or an empty message saves nothing new.
- `python -m sensai --resume` continues the last conversation, and `python -m sensai --resume 20261027-153012` a given one; the next question is answered with the previous turns as context ("And the invoice?" is understood as the invoice of the transfer discussed yesterday).
- The summary of older turns (M2) and the sources cited by each answer are restored along with the turns.
- With `SENSAI_FEATURE_SESSIONS=false`, nothing is written to disk and `--resume` is ignored with a warning.

## Keep one conversation per lead

As an investigative journalist following several leads in the same leak, I want to keep a separate conversation for each one and switch between them, so that the facts of one lead do not mix with the questions of another.

Acceptance criteria:
- `/new` starts an empty conversation and prints the id of the one that was just left, which stays saved.
- `/sessions` lists the saved conversations, most recent first, with their date, number of turns and first question, and marks the current one with `*`.
- `/resume ID` switches to another conversation without restarting Sensai; an unknown id is reported and the current conversation is kept.
- An unreadable session file is left out of the list instead of stopping Sensai.

## Never repeat my working rules

As an investigative journalist, I want Sensai to know once and for all how I work (the case, the language of the answers, what must never be written), so that every new conversation follows my rules without my repeating them.

Acceptance criteria:
- `/profile add Never write the name of the source, call it "the source".` saves the instruction in `.sensai/profile.md` (`SENSAI_PROFILE_PATH`), and the next message is already answered with it.
- At startup, Sensai says how many instructions are sent, and every conversation, new or resumed, starts with the profile after the system prompt.
- `/profile` lists the numbered instructions and `/profile remove N` deletes one; the file can also be edited by hand, one instruction per line.
- With `SENSAI_FEATURE_PROFILE=false` (or `/feature profile off`), the profile is kept on disk but not sent to the model; an unreadable profile file is reported at startup and the chat still starts.
