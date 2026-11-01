# R2 Advanced RAG: user stories

## From R1 to R2

R1 finds the chunks that look like the question; R2 filters the documents, also finds the exact terms, checks that each excerpt answers, and drops the rest.

| | R1 Basic RAG | R2 Advanced RAG |
|---|---|---|
| Documents searched | the whole index | the ones of the `/filter` (period, sender, recipient, kind, folder), also when the model searches |
| How the chunks are found | by meaning: the 5 closest vectors | by meaning and by exact words (FTS5, BM25), 30 of each, merged by their ranks (RRF) |
| How they are ranked | cosine similarity, question and chunk compared separately | a cross-encoder reads the question with each chunk and scores whether it answers |
| What reaches the model | always the 5 best, answering or not | only the chunks over the relevance threshold, then the 5 best: possibly none |
| When the documents do not answer | 5 excerpts on the same subject, that the model may answer from | the model is told that no excerpt answers, and says so |
| Text of an excerpt | one chunk of about 1200 characters | the chunk with the ones before and after it, as one passage |
| Shown to the journalist | the excerpts and their similarity | also the candidates kept and dropped (`[rerank]`), the dropped ones in `/sources dropped`, the relevance of each |
| Measured | not | `sensai eval --retrieval`, step by step |

Measured with `python -m sensai eval eval/questions.json --retrieval` on the fictional case of `eval/corpus/` (21 documents, 20 questions, 3 of them not answered by the documents), with `qwen3-embedding:0.6b` and the default cross-encoder, on a laptop CPU:

| | Recall@5 | NDCG@10 | Excerpts sent from a document that answers | Excerpts sent for a question without answer | Time per search |
|---|---|---|---|---|---|
| R1: semantic search | 91% | 0.86 | 24% | 5.0 | ~0.2 s |
| R2: + hybrid search | 97% | 0.89 | 26% | 5.0 | ~0.2 s |
| R2: + reranking | 94% | 0.95 | 25% | 5.0 | ~2.3 s |
| R2: + threshold at 20% | 91% | 0.92 | 64% | 0.7 | ~2.3 s |

The times were measured apart, on a corpus of the same size: `--retrieval` does not time the searches.

R2 sends the model the same share of the documents that answer, but they come first, and two thirds of what it reads now answers, against a quarter. For a question the documents do not answer, it reads 0.7 excerpt instead of 5 on the same subject, the ones it would make an answer from. It costs about 2 s per question, the time of the cross-encoder.

With `SENSAI_RERANK_MODEL=BAAI/bge-reranker-v2-m3`, the threshold at 20% keeps 94% of the documents that answer (NDCG 0.95), 68% of the excerpts sent answer and a question without answer gets 0.3 excerpt, but each search takes about 23 s on a CPU instead of 2.3 s.

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
