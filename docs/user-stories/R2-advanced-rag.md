# R2 Advanced RAG: user stories

## Search only the documents of a period, a person or a kind

As an investigative journalist working on a leak of several years, I want to limit the search to the documents of a period, of a sender or of a kind, so that a question on June 2023 is not answered from an email of 2021 on the same subject.

Acceptance criteria:
- `/filter after:2023-03 before:2023-07 from:dupont kind:email` limits every search of the conversation to the matching documents, `source:FOLDER/` to a folder of the index, and `/filter clear` removes the filter.
- `from:` matches the sender or the author, `to:` the recipients and copies, any part of the name, whatever its case ("élodie" finds "Élodie").
- `/filter` tells how many of the indexed documents the filter keeps.
- During the reasoning, the model can narrow the search with its own filters, but never widen the filter of the journalist.
- When no excerpt matches a search of the model, it is told which filter was applied, so that it can say so.
- The filter limits the semantic search and the keyword search alike: a document outside the filter never reaches the model.

## Find an account number, an amount or a rare name

As an investigative journalist following money, I want a question that contains an IBAN, an amount or an invoice number to find the documents that contain it exactly, so that the semantic search does not miss a term that has no meaning to it.

Acceptance criteria:
- With the `hybrid` feature on, each question is searched both by meaning (the vectors) and by its exact words (SQLite FTS5, BM25), and the two rankings are merged by Reciprocal Rank Fusion.
- A word of the title, the sender, the recipients or the attachments of an email is found too, but counts less than a word of its body.
- An index built before the keyword search gets its words when it is opened, without being rebuilt.
- `/feature hybrid off` goes back to the semantic search alone, during the chat.

## Get the excerpts that answer, not the ones on the same subject

As an investigative journalist asking about the transfer of March, I want the excerpts that answer the question to come first, and the ones that only share its words to never reach the model, so that the answer does not mix up the transfer of March with the one of May.

Acceptance criteria:
- With the `rerank` feature on, a cross-encoder (`SENSAI_RERANK_MODEL`) reads the question with each of the `SENSAI_RAG_CANDIDATES` best chunks and gives it a relevance from 0 to 1, and the chunks are sorted by it.
- A chunk under `SENSAI_RERANK_THRESHOLD` percent of relevance is dropped: it is never sent to the model.
- After each search, `[rerank] 30 candidates: 5 kept, 16 left out after the 5 best, 9 dropped under 20% relevance` tells what happened to the candidates.
- `/sources dropped` shows the excerpts dropped for the last message with their relevance, so that the journalist can check that nothing useful was thrown away; `/sources` shows the relevance of the excerpts kept.
- When every excerpt is dropped, the model is told that the documents do not answer, and says so instead of answering from what it knows; a greeting is still answered normally.
- The model is downloaded once, at the first search, and loaded only when a search needs it.

## Read the whole passage around an excerpt

As an investigative journalist, I want the model to read the sentences just before and after the chunk that was found, so that a sentence cut by the chunking does not hide who signed or when.

Acceptance criteria:
- Each excerpt kept comes with the `SENSAI_RAG_NEIGHBORS` chunks before and after it in the same document (1 by default, 0 for the chunk alone), without the overlap between chunks repeated.
- Two excerpts kept close to each other in a document become one passage, cited by one number, with the location of the whole passage (`report.pdf, p. 2-4`).
- The search itself still compares the question with small chunks, so that it stays precise.

## Prove that the search got better, and choose the threshold on data

As a member of the team presenting Sensai, I want to measure the search alone on questions whose answer is known, step by step, so that each step of R2 is kept for what it brings and the threshold is chosen on figures instead of guessed.

Acceptance criteria:
- `python -m sensai eval questions.json --retrieval` reads the questions of `sensai eval`, with the documents that answer each one in `"sources"` (none for a question the documents do not answer).
- For the semantic search alone (R1), then with the keyword search, the reranker and the threshold, it prints the Recall@k (`SENSAI_RAG_TOP_K`), the NDCG@10, the share of the excerpts sent that come from a document that answers, and the excerpts sent for the questions without answer.
- The same scores are printed for thresholds from 10 to 50%, from a single run of the reranker, and the line of `SENSAI_RERANK_THRESHOLD` is marked.
- The chat model is not called: the evaluation only needs the index, the embedding model and the reranker.
