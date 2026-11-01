# X4 Questions & Forms: user stories

## Be asked instead of getting an answer at random

As an investigative journalist whose documents name two people Vidal, I want Sensai to ask me which one I mean when my question is ambiguous, so that I do not get the amount of the wrong person.

Acceptance criteria:
- With the `questions` feature on (the default) and a model that supports tools, the model can call `ask_user` with a question and, when there are only a few possible answers, the options.
- The CLI shows `[ask_user]` and the question, then the options numbered; the call itself is not shown, so that the question appears once.
- The web page shows the question in a card, with a button for each option and a field for my own words; the call itself is not shown there either.
- The answer goes back to the model, which goes on with it, during a plain answer as during the step-by-step reasoning.
- An answer that called `ask_user` is not kept by the semantic cache.

## Answer quickly

As an investigative journalist in the middle of an interview, I want to answer the model's question in one keystroke when I can, so that a question does not slow me down.

Acceptance criteria:
- A number picks the option with that number; any other text is sent as it is, so that I can answer something the options did not offer.
- Without options, the answer is free.
- An empty answer asks the question again: nothing is answered by default.

## Keep control

As an investigative journalist, I want to be able to leave a question unanswered or to stop the questions, so that the model never blocks on me.

Acceptance criteria:
- `Ctrl+D` leaves the question unanswered: the model is told that I did not answer and not to ask again, and goes on with what it knows or says what is missing.
- In the web page, `Don't answer` does the same; stopping the answer or closing the page leaves the question unanswered too.
- `/feature questions off` (or `SENSAI_FEATURE_QUESTIONS=false`) removes `ask_user`, and the model answers without asking.
- A scheduled task and `python -m sensai eval` do not get `ask_user`: nobody is there to answer, so nothing waits for the keyboard.
- The question goes through the guardrails of the tool calls before it is shown, so that a protected name is redacted from it; my answer only lives in the messages of that turn: neither the record of the tool calls nor an interrupted turn keeps it.
