# X5 Scheduling: user stories

## Get a report every morning without asking for it

As an investigative journalist following a case for months, I want Sensai to go through my question at set times on its own, so that the morning report is ready when I arrive, without my opening a chat.

Acceptance criteria:
- `/schedule add daily 08:00 What is new about Orion Delta in the documents?` adds a task and prints its first run; `every 6h` and `weekly mon 08:00` work too, and an invalid time is refused with the accepted forms.
- `python -m sensai schedule` stays running and runs each task when it is due, with nobody at the keyboard; `python -m sensai schedule --once` runs what is due and exits, for the task scheduler of the machine.
- A task missed while the scheduler was off runs once when it starts, not once per missed slot, and then goes back to its times.
- A report is a Markdown file in `.sensai/reports/` (`SENSAI_REPORTS_DIR`) with the question, the date, the answer and the sources it cites, so that it can be read without Sensai; `/reports` lists them and marks the unread ones, `/reports N` prints one, and the chat says at startup how many are new.
- `/schedule` lists the tasks with their next run and how the last one went, shows the late ones, and `/schedule remove N` deletes one; with `SENSAI_FEATURE_SCHEDULE=false`, the commands and the scheduler refuse to run and the tasks stay in the file.

## Be alerted only when something changes

As an investigative journalist watching a company, I want a task to compare what it finds with its last run and to stay silent when nothing moved, so that an hourly check does not bury the one report that matters.

Acceptance criteria:
- Every task keeps its own conversation: the model has its earlier reports in its context, and is told to report what changed since the last one.
- When a run finds the same excerpts and the same tool results as the last one, no report is written, the conversation does not change and `/schedule` shows "nothing new". This never depends on what the model says.
- A new excerpt, or any other tool result, gives a report. So does a run that found nothing to compare, since nothing proves that nothing changed.
- `/sessions` lists the conversation of a task and `/resume task-...` continues it to ask a question about a report, but `--resume` without an id never picks it.

## Leave a task running without giving up control

As an investigative journalist protecting a source, I want a task nobody watches to do no more than I allowed when I created it, so that it never sends a query out of the machine or writes a file on its own decision.

Acceptance criteria:
- A tool that asks for approval in the chat is refused in a scheduled run, and the model is told so, unless the task was created with `--allow`: `/schedule add every 6h --allow web_search Any new article about Orion Delta?`.
- `--allow` only accepts the tools enabled in the chat, and `/schedule` shows what each task is allowed to use.
- A report says which tools were refused or failed, and such a run always gives a report, since finding the same thing may only mean it could not look.
- The guardrails apply to scheduled runs like to the chat: a task is checked when it is added, so the name of a source is redacted in the schedule file or the task is refused, and again at each run.
- A scheduled run never reuses a cached answer, and does not change the working document of a conversation.
- When a run fails, for example because Ollama is stopped, the scheduler keeps going and tries again at the next slot; `/schedule` shows the error and the chat warns about it at startup.
