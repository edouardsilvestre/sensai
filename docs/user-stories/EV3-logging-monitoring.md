# EV3 Logging & Monitoring Dashboard: user stories

## Find out why Sensai got slower

As the technical lead of a newsroom, I want to see how long the answers take day after day and prompt after prompt, so that I notice when a change to a prompt or a persona slowed Sensai down.

Acceptance criteria:
- With the `logging` feature on (the default), every message is logged in `SENSAI_LOG_PATH`, with its date, its model, its persona, the version of its prompt, its outcome, its time, the time of its first word and its tokens.
- `/stats [DAYS]` and `python -m sensai stats --days N` show the messages of the period, the error rate, the average time and the p95 of the answers of the model, the tokens per answer, and the detail by day, by prompt version, by model and by persona.
- A prompt version whose answers are at least 50% slower than those of the previous version, on the same model and with at least three answers on each side, is marked `[drift]` with how much slower it is.
- `python -m sensai stats` reads the log without starting the chat nor Ollama.

## Check what the model was given for a doubtful answer

As an investigative journalist who got a doubtful answer, I want to find the exact prompt the model received, so that I can tell whether the error came from the excerpts, the history or the model.

Acceptance criteria:
- Each message logs the question, the answer and the prompt sent: the exact messages given to the model, with the system prompt, the history, the excerpts and the tool results.
- It also logs the tools called, the excerpts found, the guardrails hit and the steps of the reasoning; the agents of `/review` are logged each with its persona.
- A message that fails (Ollama not running, Ctrl+C) is logged with the error, then the error is shown as before.
- Each message is logged with the conversation it belongs to, the same id as `/sessions`, from the first message on.

## Keep the log under control

As an investigative journalist protecting a source, I want the log to follow the same rules as the answers, so that it never becomes a file where the name of my source can be read.

Acceptance criteria:
- The question, the prompt and the answer are logged after the guardrails: a name redacted on screen is redacted in the log.
- `SENSAI_LOG_CONTENT=false` keeps only the measures, without any text; `/stats clear` empties the log; `/feature logging off` stops logging.
- The log is a file on the machine: nothing is sent to a server. A log that cannot be opened or written is a warning, and the answer is given anyway.
