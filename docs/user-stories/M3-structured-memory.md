# M3 Structured Memory: user stories

## Remember what I learn about the people and companies in the leak

As an investigative journalist, I want Sensai to record the facts I establish about a person or a company, so that I do not have to repeat them at every session.

Acceptance criteria:
- With `SENSAI_FEATURE_MEMORY=true` (the default), saying "Alvor Consulting est une société immatriculée à Chypre en 2019" makes the model call `memory_add_fact`, and the CLI prints the call (`[memory_add_fact] entity="Alvor Consulting" ...`).
- The entity is created the first time it is mentioned, with its type (person, company, place...) when the model knows it.
- The facts are saved in `.sensai/memory.db` (or `SENSAI_MEMORY_PATH`), separately from the document index.

## Find them again in a later session

As an investigative journalist, I want Sensai to recall what it knows about an entity when I ask about it, so that my investigation builds up over the days instead of starting over.

Acceptance criteria:
- After restarting Sensai, asking "Que sait-on d'Alvor Consulting ?" makes the model call `memory_search`, and the answer uses the stored facts.
- The search ignores the case and matches part of a name ("alvor" finds "Alvor Consulting").
- When nothing is stored, the model says so instead of making facts up.

## Correct or remove a fact that turned out to be wrong

As an investigative journalist, I want to correct or remove a fact from the memory, so that a wrong lead does not end up in my article.

Acceptance criteria:
- Saying "En fait, Alvor Consulting a été immatriculée en 2018" makes the model search the entity, then call `memory_update_fact` on the existing fact instead of adding a second, contradictory one.
- Saying "Oublie que X travaille pour Alvor" makes the model call `memory_delete_fact` or `memory_unlink`.
- Updating or deleting a fact that does not exist returns an error to the model, which tells the user.

## Map the links between the actors

As an investigative journalist, I want Sensai to record how the people and companies are linked, so that I can follow the money and the chain of command.

Acceptance criteria:
- Saying "Jean Dupont dirige Alvor Consulting" makes the model call `memory_link(subject="Jean Dupont", relation="dirige", object="Alvor Consulting")`, which creates the missing entities.
- `memory_search` on either entity returns the relation.
- The same relation is not stored twice.

## Keep the memory on my machine and under my control

As an investigative journalist protecting a source, I want the memory to stay local and every write to be visible, so that nothing is stored or sent without my knowledge.

Acceptance criteria:
- The memory is a local SQLite file: no request leaves the machine to read or write it.
- Every write (add, update, delete, link, unlink) is printed in the CLI when the model makes it.
- `/feature memory off` stops the model from reading or writing the memory for the rest of the session; the file is kept.
- The memory needs `SENSAI_FEATURE_TOOLS=true`: without tools, no memory tool is sent to the model.
