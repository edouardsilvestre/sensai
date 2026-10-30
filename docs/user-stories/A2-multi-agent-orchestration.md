# A2 Multi-agent orchestration: user stories

## Get an answer checked before I read it

As an investigative journalist, I want my question answered by the investigator and then checked by the fact-checker, so that I only read claims that the documents support, without switching persona by hand.

Acceptance criteria:
- `/review QUESTION` runs the investigator, then the fact-checker on its answer; a header names each agent before it speaks.
- The fact-checker gives a verdict per claim and ends with `VERDICT: APPROVED` or `VERDICT: REJECTED`.
- Rejected, the investigator gets the fact-checker's remarks and corrects its answer, which is checked again.
- After `SENSAI_MAX_REVIEW_ROUNDS` checks (2 by default) without approval, the review stops and says that the answer is not approved; the editor does not run.
- A verdict that cannot be read counts as a rejection.

## Get a publishable text from checked findings only

As an investigative journalist, I want the approved answer handed to the editor, so that the draft I get only contains facts that passed the fact-check.

Acceptance criteria:
- Once the fact-checker approves, the editor writes the text from the question and the approved answer.
- Only the question and the final text (the editor's, or the last answer when it was not approved) are kept in the conversation history and saved with the session.
- The fact-checker and the editor cannot call tools: only the investigator can.

## Turn the review off

As the technical lead of a newsroom, I want to turn the review off, so that a team that does not use it never triggers it by mistake.

Acceptance criteria:
- `SENSAI_FEATURE_ORCHESTRATION=false` in `.env`, or `/feature orchestration off` in the chat, turns it off; `/features` shows its state.
- Turned off, `/review` is refused with the command to turn it on; plain messages are answered as before, by a single agent.
- `/review` without the `investigator`, `fact_checker` or `editor` persona is refused with the names of the missing ones.
