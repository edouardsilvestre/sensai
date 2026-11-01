# X2 Branching, Resume & Interrupt: user stories

## Try another question without losing the first answer

As an investigative journalist testing a lead, I want to change a question I asked earlier and get a new answer from that point, so that I can explore another direction and still find the first conversation intact.

Acceptance criteria:
- `/history` lists the messages of the branch with their numbers and the beginning of each answer.
- `/edit 2 Who signed the invoice?` creates a new branch from message 2, and the model answers with the history before message 2 only, not the messages that followed it.
- `/edit 2` without text shows message 2 and asks for the new one; an empty answer cancels the edit.
- The original conversation is not changed: `/branch 1` comes back to it, with all its messages.
- A message the guardrails block, or an Ollama failure, creates no branch: the CLI says it and stays on the current branch.

## Find my way between versions of a conversation

As a journalist who compares several hypotheses on the same case, I want to list the branches of a conversation and switch between them, so that each hypothesis keeps its own history.

Acceptance criteria:
- `/branches` lists each branch with its number, its number of turns, where it comes from and its first edited message; the active one is marked with `*`.
- `/branch 1` continues branch 1, and the model gets the history of that branch only.
- Branches are saved with the conversation: `--resume` and `/resume` come back to the same branch, and all the others are still there.
- `/new` starts a new conversation with one branch.

## Keep the investigation file consistent in each branch

As a journalist whose investigation file is built from the conversation, I want each branch to have its own working document, so that a hypothesis I explore never ends up in the file of another.

Acceptance criteria:
- A new branch starts from the document as it was before the edited message.
- A change to the document in a branch does not appear in any other branch, and `/document undo` undoes the changes of the active branch.
- The compaction of a branch does not change the others.

## Stop an answer that goes the wrong way

As a journalist reading an answer as it is written, I want to stop it as soon as it goes in the wrong direction and say what I want instead, so that I do not wait for a long answer I will not use.

Acceptance criteria:
- Ctrl+C during an answer stops it at once, and the CLI says what was written is kept.
- The question and the text already shown stay in the history, marked `[answer interrupted by the user]`, so that the next message ("shorter, and only the amounts") redirects the answer.
- When tools were called before the interruption, the text of the earlier rounds and the tool results stay too, after the guardrails, so that the model does not lose what it found. The same goes for the tools called while reasoning.
- An interrupted answer does not update the working document, since a cut answer establishes nothing.
- A cut `/json` answer is not kept. An interrupted `/review` keeps the question and the last text written: the text being written, or the investigator's answer while the fact-checker reads it.
- The conversation is saved after the interruption, and `/edit` can also replace the interrupted question.
