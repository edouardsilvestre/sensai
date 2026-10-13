# R1 Basic RAG: user stories

## Question a leak and get sourced answers

As an investigative journalist, I want to ask questions in plain language about the documents of a leak, so that I find the relevant passages without reading thousands of files and can check every claim against its source.

Acceptance criteria:
- After `python -m sensai ingest leak/`, a question about the documents is answered from the indexed excerpts, and each claim cites an excerpt number (`[1]`).
- Below the answer, the cited excerpts are listed with their file and, when it applies, their page (`report.pdf, p. 4`) or row (`ledger.xlsx, Transfers, rows 12-30`).
- `/sources` shows the full text of every excerpt retrieved for the last question, with its similarity score.
- When the excerpts do not contain the answer, Sensai says so instead of making one up; a citation of an excerpt that does not exist is flagged with a warning.
- No network request leaves the machine apart from the local Ollama server.

## Index a leak that arrives in several deliveries

As an investigative journalist whose source sends documents in several batches, I want to add new files to the index without re-indexing everything, so that I can keep working on the case while new documents arrive.

Acceptance criteria:
- Running `ingest` again on the same folder only processes new or modified files; unchanged files are reported as `unchanged` and are not embedded again.
- Files deleted from an ingested folder are removed from the index.
- An unreadable file (corrupted or scanned PDF, unknown type) is reported with its reason and does not stop the ingestion of the other files.
- Interrupting an ingestion with `Ctrl+C` keeps the files already indexed, and running the same command again resumes it.

## Mix heterogeneous documents in the same case

As an investigative journalist, I want PDFs, emails and spreadsheets from the same leak to be searchable together, so that one question can surface an email, the bank statement row and the report page that relate to the same transfer.

Acceptance criteria:
- `.pdf`, `.eml`, `.csv`, `.tsv`, `.xlsx`, `.html`, `.md` and `.txt` files are indexed from the same folder.
- An email excerpt comes with its sender, recipients, date and subject, so that "who wrote to whom, and when" can be answered.
- A spreadsheet row is read as `column: value` pairs, so that a row matching the question is found and cited on its own.
