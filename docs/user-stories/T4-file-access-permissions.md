# T4 File Access with Permissions: user stories

## Let the model read the files of my case

As an investigative journalist with a folder of leaked documents and notes, I want to ask Sensai to look into a given file ("lis le mail de Marc Vidal") without indexing everything first, so that I can check one document during the conversation.

Acceptance criteria:
- With `SENSAI_FEATURE_FILES=true` (or `/feature files on`) and `SENSAI_FILES_READ=leak`, the model can list `leak` (`list_files`) and read its files (`read_file`); each call and its result are printed (`[read_file] leak/mail-vidal.eml, lines 1-12 of 12`).
- PDFs, emails, spreadsheets and HTML are read as text, with the page numbers of a PDF; a binary file is refused.
- A long file is sent in parts: the model gets the next `start_line` and reads on, so that one file never fills the context window.
- The content is marked as data, not instructions: an instruction hidden in a document is not followed.
- Reading needs no approval, since nothing leaves the machine.

## Keep the model inside the folders I allow

As an investigative journalist, I want the model to touch only the folders I name, and never the files that protect my source, so that a wrong or manipulated tool call cannot read or change anything else on my machine.

Acceptance criteria:
- Every path is checked before the file is opened: a path outside `SENSAI_FILES_READ` and `SENSAI_FILES_WRITE` is refused with the reason (`secret.txt is outside the folders you may read (SENSAI_FILES_READ: leak)`), and the model is told why.
- `..`, absolute paths and links are resolved first, so `leak/../secret.txt` is refused like `secret.txt`.
- With `SENSAI_FILES_DENY=sources`, `leak/sources/contact.txt` is refused (`denied by the pattern 'sources'`) and `list_files` does not show the folder.
- `.env`, `.sensai` (the index, the sessions, the memory, the guardrail rules) and `.git` are always refused, even inside an allowed folder.
- `/files` shows the folders that may be read, the ones that may be written, those that do not exist, and the denied patterns.
- Without the feature, the model has no file tool at all.

## Save a draft where I decide

As an investigative journalist, I want Sensai to write its drafts and notes as files in a folder I choose, after I have seen them, so that I get a file I can edit instead of copying an answer from the terminal.

Acceptance criteria:
- `write_file` creates or replaces a text file, and the missing folders, only inside `SENSAI_FILES_WRITE`; by default no folder may be written.
- Before writing, the CLI shows the path and the full content and asks: `Allow write_file? It writes a file on this machine.`; refused, nothing is written and the model is told.
- The content goes through the guardrails first: with a rule protecting `Claire Benoit`, the file contains `[SOURCE]` instead of her name.
- A file is written next to its destination then renamed, so that a failure never leaves half a file; there is no tool to delete a file.
