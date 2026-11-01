# Sensai

**Sensai is an offline AI copilot for investigative journalists.**

A journalist receives a leak of 5,000 internal documents from a company. She can't upload a single file to the cloud without putting her source at risk. Sensai runs entirely on an air-gapped machine: it indexes the documents, identifies the people, companies and money flows involved, builds a timeline of events, and flags contradictions between documents.

Everything stays local, because protecting sources is non-negotiable. And every claim Sensai makes is tied back to the exact document and passage it comes from: in investigative journalism, an unsourced statement is a defamation risk. Faithfulness to sources isn't a nice-to-have here, it's the core of the product.

### Key features

- 🖥️ **Web UI**: `python -m sensai web` opens the same conversations in a simple page on this machine, with streaming, the context used, the personas, the features and approval buttons, and nothing loaded from the internet.
- 🔒 **Fully offline by default**: no network access, no cloud, no data ever leaves the machine. Web search and public links are opt-in, and every query or page is shown before it leaves.
- 💾 **Sessions & profile**: every conversation is saved and can be resumed days later, and a profile of standing instructions (the case, the language, what must never be written) is given to the model at the start of every conversation.
- 🌿 **Branches & interruption**: `/edit` tries another question from any point of the conversation without losing the first version, and Ctrl+C stops an answer to redirect it.
- 🛡️ **Source & privacy guardrails**: the name of the source is removed from every answer and web search, personal data never leaves the machine, and requests targeting someone's private life are refused, following rules set in a file.
- 🌐 **Web search**: when enabled, the model can look up public facts (companies, people, news) and cites the URLs it used.
- 🔌 **MCP servers**: the model can call the tools of any MCP server reached by its URL, like a company registry lookup, each call approved first.
- 🔎 **Advanced RAG**: semantic and keyword search across thousands of heterogeneous documents (PDFs, emails, spreadsheets...), filtered by date, sender or kind, reranked so that only the excerpts that answer reach the model, and every answer cited to its source.
- 🕸️ **Entity graph memory**: the model saves what is established about people, companies and places, and who is linked to whom, in a local database it reads and corrects across sessions.
- 📅 **Timeline & contradiction detection**: events are placed on a chronology, and inconsistencies between documents are surfaced automatically.
- 📂 **Living investigation file**: a case document, apart from the chat, that the model updates section by section after every answer with the facts established and their source, and that can be undone and exported.
- 📤 **Export & publication**: the conversation and the document go to Markdown, PDF, HTML or JSON, and, when turned on, to a frozen public page, checked by the guardrails and served apart from the chat, that a colleague opens without an account and that one command takes down.
- 📊 **Code execution for financial analysis**: Sensai writes and runs code to analyze spreadsheets and trace money flows.
- 🧾 **Structured outputs**: `/json` answers in JSON that follows a fixed schema (the answer, its claims with their excerpts, the people and companies, the events), ready for a database or another tool.
- 🧠 **Traceable reasoning**: when turned on, Sensai reasons step by step (what it knows, what is missing, which document or tool to check next) and every step can be read again with `/reasoning`, so a conclusion can be checked before it is published.
- ♻️ **Semantic cache**: a question close to one already answered, in the same context, can get the same answer instantly once you confirm it, without calling the model again.
- 💡 **Answer examples**: keep your best answers with `/example add`, and Sensai shows the closest ones to the model for each new question, so that even a small model cites its excerpts and says what the documents do not say.
- ❓ **Questions to the user**: when a request is ambiguous, the model can ask the journalist a question, with a few options to pick from, and goes on with the answer instead of guessing.
- ⏰ **Scheduled tasks**: a question Sensai goes through on its own every morning or every hour, comparing with its last run, and writing a sourced report only when it finds something else, with no tool used unless the journalist allowed it in advance.
- 🗂️ **Prompt versioning**: each change to the system prompt is kept as a dated version that can be compared, measured on your own test questions, and rolled back with `/prompt use N`.
- 📈 **Logs & metrics**: every message is logged on the machine with the prompt sent, the answer, the tools, the time and the tokens, and `/stats` shows the volume, the response time, the error rate and the prompt changes that slowed Sensai down.
- 🎯 **Adversarial testing**: audit resilience against jailbreaks, prompt injections and malformed inputs with
  `python -m sensai attack`, track defense rates, and detect regressions.
- 🛡️ **Hallucination detection**: each generated statement is checked against the source documents; anything unsupported is flagged instead of presented as fact.
## How does it work?

`sensai/core` runs the conversation and emits events, `sensai/interfaces` displays them. Only `sensai/core/llm/ollama.py` talks to Ollama.

### Document search (RAG)

Indexing (`python -m sensai ingest`):

1. `rag/loaders.py` extracts the text of PDF, eml, csv, tsv, xlsx, html, md and txt files, with page numbers, sheet and row numbers, and email headers.
2. `rag/chunker.py` splits the text into sentences or rows and groups them into chunks of about 1200 characters, with 200 characters of overlap.
3. `rag/embedder.py` computes one vector per chunk with Ollama (`/api/embed`).
4. `rag/store.py` saves documents, chunks and vectors in one SQLite file, the vectors in a [sqlite-vec](https://github.com/asg017/sqlite-vec) table.

Files that did not change are skipped on the next run, and deleted files are removed from the index. An index made before the search filters (format v1) is refused with a message: rebuild it with `ingest --rebuild`. An index made before the neighbors gets the `overlap` column when it is opened, at 0: its passages repeat a few sentences between chunks until it is rebuilt.

Answering (`DocumentIndex.search`), for each message:

1. **Filters** (`/filter`, or the arguments of `search_documents` during the reasoning): `store.allowed_documents` keeps the documents of the period, the sender, the recipient, the kind or the folder asked, from the `date`, `author`, `sender` and `recipients` columns copied from the metadata at indexing. Both searches below only look at these documents, inside their top-k, so that a filter never leaks a document nor leaves fewer results than it could.
2. **Hybrid search** (`hybrid` feature): the `SENSAI_RAG_CANDIDATES` (30) chunks closest to the message by cosine similarity, and the 30 that match its words best in a SQLite [FTS5](https://www.sqlite.org/fts5.html) table by BM25, where a rare word counts more and the header of an email (title, people, attachments) counts half. The two rankings cannot be compared by their scores: `retriever.fuse` merges them by their ranks only ([Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)), and a chunk found by both comes first. The vectors find "the suspicious payments", the keywords find "CH93 0076 2011".
3. **Reranking** (`rerank` feature): the vectors compare the question and a chunk separately, so a chunk on the same subject that does not answer scores as high as the one that does. `rag/reranker.py` reads the question and each of the 30 chunks together with a cross-encoder ([sentence-transformers](https://sbert.net/docs/cross_encoder/usage/usage.html), `SENSAI_RERANK_MODEL`) and gives it a relevance from 0 to 1. It is too slow to read the whole index, hence the fast search first. The raw score of the model goes through a sigmoid, so that the threshold means the same with any model.
4. **Threshold**: a chunk under `SENSAI_RERANK_THRESHOLD` percent of relevance is dropped and never reaches the model, like the transfer of May for a question on March; then the `SENSAI_RAG_TOP_K` best are kept. When every chunk is dropped, the model is told that the documents do not answer, instead of getting the question alone and answering from what it knows.
5. **Neighbors**: a chunk is cut every 1200 characters, and the sentence that answers can be just before the cut. Each excerpt kept comes with the `SENSAI_RAG_NEIGHBORS` chunks before and after it in the same document, joined without their overlap: the chunker records how many characters each chunk repeats from the one before (`chunks.overlap`), and `chunker.join_chunks` skips them. Two excerpts close in a document become one passage. The search stays precise on small chunks, the model reads the whole passage.

The excerpts are added to the prompt, numbered, and the model cites them as `[1]`, `[2]`. Only the message is kept in the history, not the excerpts. After each search, the CLI tells what happened to the candidates, and `/sources dropped` shows those under the threshold:

```
[rerank] 30 candidates: 5 kept, 16 left out after the 5 best, 9 dropped under 20% relevance (/sources dropped)
[search] 5 excerpts from 4 documents
```

Measured with `python -m sensai eval eval/questions.json --retrieval` on the 21 documents of a fictional case (`eval/corpus/`), with 20 questions of which 3 are not answered by the documents, `qwen3-embedding:0.6b` and the default reranker:

| Step | Recall@5 | NDCG@10 | Excerpts sent from a document that answers | Excerpts sent for a question without answer |
|---|---|---|---|---|
| Semantic search (R1) | 91% | 0.86 | 24% | 5.0 |
| + hybrid search | 97% | 0.89 | 26% | 5.0 |
| + reranking | 94% | 0.95 | 25% | 5.0 |
| + threshold at 20% | 91% | 0.92 | 64% | 0.7 |

The keyword search finds the IBAN and the invoice number the vectors missed. The reranker puts the document that answers first (NDCG 0.89 to 0.95), but pushes one out of the 5 best. The threshold does the most for the model: for a question the documents do not answer, it gets 0.7 excerpt instead of 5 off-topic ones, and two thirds of the excerpts it reads come from a document that answers, against a quarter, for the recall of the semantic search alone. It costs a few documents that answer, mostly spreadsheet rows, which the cross-encoder reads poorly.

The threshold was chosen from the scores the same command prints for thresholds from 10 to 50%:

| Threshold | Recall@5 | NDCG@10 | Excerpts sent from a document that answers | Excerpts sent for a question without answer |
|---|---|---|---|---|
| 10% | 91% | 0.92 | 57% | 0.7 |
| **20%** | **91%** | **0.92** | **64%** | **0.7** |
| 30% | 88% | 0.90 | 65% | 0.7 |
| 50% | 82% | 0.83 | 68% | 0.3 |

Up to 20%, the noise falls from 5 excerpts to 0.7 for the loss of one document that answers; past it, each step loses documents that answer for a few points of precision. Twenty questions on documents written for the test are a small sample: run the evaluation on your own case before changing the threshold.

The default cross-encoder, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, is multilingual and reads 30 candidates in about 2 s on a laptop CPU, plus about 5 s to load at the first search. `BAAI/bge-reranker-v2-m3` (`SENSAI_RERANK_MODEL=BAAI/bge-reranker-v2-m3`) does better on the same evaluation, but takes about 23 s per search without a GPU:

| Cross-encoder | + reranking: Recall@5 / NDCG@10 | + threshold at 20%: Recall@5 / NDCG@10 | Excerpts sent from a document that answers | Excerpts sent for a question without answer | Time per search |
|---|---|---|---|---|---|
| `mmarco-mMiniLMv2-L12-H384-v1` (default) | 94% / 0.95 | 91% / 0.92 | 64% | 0.7 | ~2.3 s |
| `bge-reranker-v2-m3` | 97% / 0.97 | 94% / 0.95 | 68% | 0.3 | ~23 s |

It is worth it with a GPU, or for a batch of questions that can wait; for a chat on a CPU, the default model answers ten times faster.

### Sessions & profile

- `memory/sessions.py`: `SessionStore` writes each conversation to its own JSON file in `SENSAI_SESSIONS_DIR` after every answered message: the summary of older turns and the turns kept verbatim, with the sources each answer cited and a short record of its tool calls, exactly as `ConversationMemory` holds them. Resuming puts them back in the memory, so the model gets the same context as before the restart. A file is written next to the session then renamed, so a crash never leaves half a conversation, and an unreadable file is skipped instead of stopping Sensai.
- `memory/profile.py`: `Profile` reads `SENSAI_PROFILE_PATH`, a plain text file with one instruction per line, that `/profile add` and `/profile remove` edit and that can also be edited by hand. `ConversationMemory` sends it in the system message, after the system prompt and before the summary, so every conversation starts with it, new or resumed, and it is never condensed by the compaction.

Sessions and the profile are two features (`SENSAI_FEATURE_SESSIONS`, `SENSAI_FEATURE_PROFILE`). Like the index, they stay in `.sensai/` on the machine: delete `.sensai/sessions/` to erase every saved conversation.

### Branches & interruption

A conversation is not a straight line: the journalist often wants to ask a question differently without losing what the first version gave, or to stop an answer that goes the wrong way.

- `ConversationMemory` (`memory/conversation.py`) holds several `Branch`es. `/edit N` calls `fork`: the new branch gets the turns before message N, the summary, and the working document as it was before that turn, which each `Turn` records. Each branch has its own copy of its history and of its document, so that an answer, a compaction or a document edit in one branch never reaches another. The cost is some repeated text in the session file; the older turns cannot be edited, since the compaction has replaced them with the summary.
- `SessionStore` saves the active branch as before, and the others in `other_branches`: older session files are read as a conversation with one branch.
- An edit that gets no answer (blocked by a guardrail, or Ollama failed) removes its branch instead of leaving an empty one.
- Ctrl+C stops the generator of `Agent.run`, which closes the connection, so Ollama stops generating too. `Agent.interrupt` then adds the question and the text already shown to the history, followed by `[answer interrupted by the user]`, so that the next message redirects the answer: the model sees where it was stopped. Only the text that passed the guardrails is kept, along with the earlier text and the tool results of the turn, including those of the reasoning, which go through the same rules, so that the model can go on from there. It stays out of the working document, a cut `/json` answer is not kept, and an interrupted `/review` keeps the question and the last text written.

It is the `branching` feature, on by default: off, the commands are refused and an interrupted turn is dropped, as a stopped generation has nothing to keep.

### Long-term memory

Sessions keep conversations; the long-term memory keeps what they established, as structured records the model reads and writes itself with tools:

- `memory/store.py`: `MemoryStore` holds three SQLite tables in `SENSAI_MEMORY_PATH`: `entities` (a person, a company, a place, with its type), `facts` (one sentence about an entity) and `relations` (`Jean Dupont` `runs` `Alvor Consulting`). Names and relations ignore case, so the same entity or relation is never stored twice, and deleting an entity deletes its facts and relations.
- `tools/memory.py`: `MemoryTools` gives the model six tools for the CRUD operations:

| Tool | What it does |
|---|---|
| `memory_search(query)` | the entities whose name or one of whose facts contains the query, with the id of each fact and the relations |
| `memory_add_fact(entity, fact, entity_type)` | saves a fact, creating the entity if it is new |
| `memory_update_fact(fact_id, fact)` | replaces a fact that turned out to be wrong |
| `memory_delete_fact(fact_id)` | deletes a fact |
| `memory_link(subject, relation, object)` | saves a relation, creating the missing entities |
| `memory_unlink(subject, relation, object)` | deletes a relation |

Every call is printed in the CLI (`[memory_add_fact] fact 3 saved`), so the journalist sees each write. The search results are marked as data, not instructions: a fact copied from a web page cannot give orders to the model in a later session. The memory never leaves the machine, so its tools need no approval; delete `.sensai/memory.db` to erase it.

It is the `memory` feature, on by default, and needs a model that supports tools. Small models like `qwen3:1.7b` only call the tools reliably with their thinking phase on: leave `SENSAI_THINK` empty or `true`.

### Working document

The chat is a transcript; the working document is the deliverable. `ConversationMemory` holds a `Document` (`document/document.py`), a Markdown file made of `## ` sections, by default `Summary`, `People and companies`, `Timeline` and `Open questions`. It is never condensed by the compaction, and it is saved with the conversation and resumed with it.

After each answer, `document/editor.py` asks the chat model, in a call of its own, how the exchange changes the document. The model never rewrites the document: it answers with edits, and Ollama's structured outputs (`format`, a JSON schema) make sure the reply is always `{"edits": [...]}`:

| Edit | What it does |
|---|---|
| `{"op": "append", "section", "text"}` | adds lines at the end of a section |
| `{"op": "replace", "section", "find", "text"}` | corrects a passage, which must be found exactly once in that section |
| `{"op": "rewrite", "section", "text"}` | replaces one section |
| `{"op": "add_section", "section", "text"}` | creates a section |

`Document.apply` checks each edit on its own and rejects it with the reason instead of guessing: an unknown section, a passage to correct that is not found or found twice, a line the section already has (whatever the way its source is cited), more than 8 edits in one turn. A `## ` line in a text is turned into `### `, so an edit can never split the document. The valid edits are applied, the others are printed as warnings, and an invalid reply leaves the document as it was.

The edits go through the guardrails like an answer: the source's name is redacted in the document too, and an edit with blocked data is not written. The document is sent to the model after the profile, so that it can answer "what is still missing?", and `/document undo` puts back the document as it was before the last answer.

It is the `document` feature, on by default. It costs one more call to the model after each answer, and the edits are only as good as the model: small models often fail to quote the passage to correct, and their correction is then rejected instead of applied.

### Personas

A persona sets the agent's name, role, tone and scope: it is the system message of the conversation. Each `personas/<id>.json` file defines one, so a newsroom can add its own profile without touching the code. Sensai starts without persona unless `SENSAI_PERSONA` or `-p ID` picks one, and ships three personas that follow an investigation:

| Persona | Role |
|---|---|
| `investigator` | finds leads, links people, companies and money flows, builds the timeline |
| `fact_checker` | challenges every claim: supported, unsupported or contradicted |
| `editor` | turns the established findings into publishable prose, without adding facts |

```json
{
  "name": "Sensai Legal Reviewer",
  "role": "a media lawyer reviewing an article before publication.",
  "tone": "cautious and precise.",
  "scope": "flag defamation and privacy risks; do not rewrite the article.",
  "instructions": "Optional extra instructions."
}
```

`name`, `role`, `tone` and `scope` are required; a file with a missing field or invalid JSON stops the startup with an error naming the file. Switching persona (`/persona ID`) only replaces the system message: the turns and the summary of older turns are kept, so the fact-checker knows what the investigator established.

The scope is sent as work that can be asked for (`When the user asks for it, you can help with this work: ...`), after `Answer the user's message and nothing else.`, and not as a `Scope:` line. With `Scope:`, `granite3.1-moe` took it for a task and started by itself: asked to just say OK, the investigator went on to analyse leaked documents it had never seen. Measured on six tries for each case, the answers to that message at the start of a conversation became about twice as short, and an ordinary question went from 853 to 172 characters, while the answers on a real excerpt still named the right person six times out of six. It is not enough for this model: it still does not stop at OK, and after a change of persona it imitates the form of its earlier answers, since the history is kept. A rule in the prompt, a reminder just before the message and turning the history into a note changed nothing. The model matters more than the prompt: in the same conversation, `qwen2.5:1.5b`, half the size on disk (1 GB against 2 GB), said OK six times out of six without borrowing anything from the earlier answers, and it really calls tools, where `granite3.1-moe` writes its calls as text.

### Prompt versions

With the `prompt_versions` feature (on by default), the system prompt is kept at every start in `SENSAI_PROMPTS_DIR/system.json`: a system prompt changed in configuration becomes a new dated version, and the version the agent uses from then on.

```
System prompt changed: kept as version 2. /prompts to list them.
> /prompts
Versions of the system prompt (* in use):
  * 2. 2026-11-01 14:30  9c1e04ab  262 chars
    1. 2026-10-12 09:12  60f1efc2  248 chars
> /prompt diff 1 2
> /prompt use 1
Sensai uses version 1 of its system prompt from now on.
```

`/prompt show N` prints a version, `/prompt diff N M` what changed between two, and `/prompt use N` rolls back to version N: the agent uses it from then on, and after a restart too, until the configured prompt changes again. Each saved conversation records the version of its prompt (`prompt_version` in the session file), so an answer can be traced back to the prompt that wrote it.

`/prompt compare N M` asks the questions of `SENSAI_PROMPT_CASES_PATH` (see `prompt_cases.example.json`) with each version, and prints how many answers pass their checks, the average time and the average tokens per answer:

```
Version  Passed   Avg time  Avg tokens
1        2/3      4.12      412
2        3/3      3.05      298
  version 1, case 1 (Écris un titre : Marc Vidal a signé...): missing "480 000"
```

A case is a `question`, and optionally the texts its answer must contain (`must_contain`) and must not contain (`must_not_contain`), ignoring the case. To compare only the prompts, each question is asked alone with the system prompt, without the history, the documents nor the tools, at temperature 0 with a fixed seed. The result of each comparison is kept with the versions, and `/prompts` shows the last one. Turning the feature off goes back to the configured system prompt, and keeps the versions.

### Multi-agent review

`/review QUESTION` has the three personas work on the same question, one after the other, instead of switching them by hand:

1. The **investigator** answers, with the documents and the tools.
2. The **fact-checker** gets the question and the answer, answers in JSON, following a schema sent to Ollama (`{"claims": [{"claim", "reason", "verdict"}]}`), and the answer is approved only when every claim is `supported` or `common knowledge`. Rejected, the investigator gets its remarks, corrects its answer and the fact-checker checks it again, up to `SENSAI_MAX_REVIEW_ROUNDS` checks (2 by default).
3. Once approved, the **editor** writes the text from the checked answer. Never approved, the review stops without the editor and says so.

`orchestrator.py`: `Review` builds one `Agent` per persona and passes the text from one to the next. Each agent has its own memory: the fact-checker and the editor only get the text they work on, so the reasoning of the investigator does not sway them, and only the investigator can search the documents and call tools. The investigator starts from the conversation, so a follow-up question is understood. The guardrails, the approvals and the profile apply to every agent.

Only the question and the final text (the editor's, or the last answer when it was not approved) are added to the conversation and saved with the session. The investigator reads the working document, but the review agents never edit it: the document is updated once, from the editor's text, and only when the fact-checker approved it. A verdict that cannot be read, or that lists no claim, counts as a rejection, so an unclear fact-checker never lets an answer through. It is the `orchestration` feature, on by default; plain messages are still answered by a single agent.

### Structured outputs

A prose answer is for the journalist; a program needs data it can read without guessing. Sensai sends a JSON schema to Ollama as `format` ([structured outputs](https://docs.ollama.com/capabilities/structured-outputs)): the model can only write JSON that follows it, instead of free text. Three places use it:

| Where | Schema | What reads it |
|---|---|---|
| `/json QUESTION` | `ANSWER_SCHEMA` (`structured.py`) | the journalist, or any tool the JSON is copied into |
| the working document | `EDITS_SCHEMA` (`document/editor.py`) | `Document.apply`, which applies the edits |
| `/review` | `VERDICT_SCHEMA` (`orchestrator.py`) | `approved`, which computes the approval from the verdicts |

`/json` answers like a message, with the documents, the profile and the guardrails, but in this form:

```json
{
  "answer": "Orion a payé 2M à Delta le 4 avril.",
  "claims": [{"text": "Orion a payé 2M à Delta", "sources": [1, 3]}],
  "entities": [{"name": "Orion", "type": "company"}, {"name": "Delta", "type": "company"}],
  "events": [{"date": "2024-04-04", "description": "Virement de 2M d'Orion à Delta"}]
}
```

`sources` are the numbers of the excerpts listed below the JSON, as `[1]` in a prose answer, and an `entities` type is one of `person`, `company`, `place` or `other`. The JSON is printed once complete, indented, after being checked by `parse_answer` against the schema: Ollama already constrains the reply, but a guardrail can cut it and another server could ignore `format`. A reply that does not follow the schema is printed as it is, with what is wrong; an excerpt number that does not exist is flagged, like in a prose answer.

The schema constrains every token the model writes, so it could not call a tool: an answer with a schema is given no tool (`Agent.run`), only the excerpts of the documents. The order of the fields matters too, since the model writes them in that order: the fact-checker gives the `reason` of each verdict before the `verdict`, so that it decides after having reasoned.

### Reasoning

With `/feature reasoning on` (or `SENSAI_FEATURE_REASONING=true`), Sensai reasons step by step before answering, and shows each step: what it knows, what is missing, and the document search or the tool it uses next. It is off by default, because each step is a call to the model.

```
> /feature reasoning on
reasoning is on.
> Qui dirige Orion Delta Consulting, et cette personne a-t-elle reçu de l'argent de Nordbank ?
[search] 3 excerpts from 3 documents
[thought 1] The first excerpt contains the information about the legal structure of Orion Delta Consulting, and the second excerpt mentions a specific transaction involving Marc Vidal [...]
[search_documents] query="Orion Delta Consulting gérant"
[search_documents] 3 excerpts: docs/statuts_orion.txt, docs/releve_nordbank_2023-04.txt, docs/meteo.txt
[thought 2] [...] it states that the company is governed by Marc Vidal with no term limit. [...] it mentions a transfer of 480,000 EUR to Marc Vidal's personal account on April 5, 2023 [...]
Les documents indiquent que la société Orion Delta Consulting est gouvernée par Marc Vidal. L'excerpt [1] mentionne que Marc Vidal est nommé gérant de la société pour une durée illimitée. L'extrait [2] cite un virement de 480 000 EUR vers son compte personnel sur la banque Norbahn le 5 avril 2023, libellé "conseil stratégique".

> /reasoning 1
Question 1: Qui dirige Orion Delta Consulting, et cette personne a-t-elle reçu de l'argent de Nordbank ?
[thought 1] ...
  [search_documents] query="Orion Delta Consulting gérant"
    <excerpt source="docs/statuts_orion.txt">
    [...] La société est gérée par M. Marc Vidal, nommé gérant pour une durée illimitée.
    [...]
```

`/reasoning` lists the questions of the conversation, `/reasoning N` shows the thoughts, the tool calls and the full results of question N, so that the journalist can check what a conclusion rests on. The reasoning is never sent back to the model: only the question, the answer and a short record of the tool calls are kept in the history, plus the tool results of a turn the user interrupts. `/new` and `/resume` clear it.

`reasoning.py` and `Agent._reason`: at each step, the model replies with a JSON object `{"thought", "action", "arguments"}` that Ollama is forced to follow (the `format` option of the chat API). The schema has one form per tool, with the arguments of that tool, plus `answer`: the model cannot invent a tool or an argument. A tool chosen by the model goes through the guardrails and the approval like any tool call, and its result is given back to the model for the next step; the thoughts go through the output guardrails before they are shown. Once the model picks `answer`, the final answer is streamed from the steps and their results, without tools. With indexed documents and the `rag` feature on, the reasoning can also call `search_documents`, which searches the documents again with the model's own words to follow a lead (the manager found in the statutes, then his transfers). The excerpts it finds are numbered after those of the first search, so that the answer can cite them, and they are listed with the sources and in `/sources`. As the actions come from the schema, the reasoning also works with models that cannot call tools, like `llama3`.

The reasoning stops after `SENSAI_MAX_TOOL_ROUNDS` steps, or on a step that is not valid, and Sensai answers from what it found. Its quality depends on the model: a small model like `qwen2.5:1.5b` may write its thoughts in English, stop before following every lead, or misspell a name it copies (Nordbank became "Norbahn" in the answer above). The agents of `/review` show their thoughts but do not add them to `/reasoning`. An answer that follows a schema of its own, like `/json` or the verdict of the fact-checker, is given without reasoning.

### Semantic cache

A journalist often asks the same question again in other words, days later, in a new conversation. With the `cache` feature (on by default), Sensai keeps each answer with its question, and when a new question is close to one already answered, it offers the answer it kept:

```
> Quelle est la capitale de l'Australie ?
La capitale de l'Australie est Sydney.

> /new
> Quelle ville est la capitale de l'Australie ?
[cache] A close question was answered on 2026-10-31 at 18:07: "Quelle est la capitale de l'Australie ?" (similarity 99%)
Use this answer? [y]es / [n]o: y
[cache] Answer reused, the model was not called.
La capitale de l'Australie est Sydney.
```

With `qwen2.5:1.5b`, an answer took 8.1 seconds and its reuse 0.6 seconds, the time to embed the new question. The answer above is also wrong, and the cache served it again as it was: the cache saves the model's time, it does not check what it says. `n` lets the model answer again, and `/cache clear` removes every answer kept.

`cache.py`: `SemanticCache` embeds the question with `SENSAI_EMBED_MODEL` and keeps the vector, the answer, its sources and its date in a SQLite file at `SENSAI_CACHE_PATH`. A new question is compared to the questions kept in the same context by cosine similarity, and the closest one is offered when it reaches `SENSAI_CACHE_THRESHOLD` percent. The context is a hash of the model, the conversation sent to the model (the system prompt or the persona, the profile, the earlier turns), the indexed documents, the guardrail rules, the answers kept as examples shown to the model and the `rag`, `tools` and `reasoning` features: a follow-up question, another persona, a changed document, a new rule or a new example is another context, and is never answered from the cache. A cached answer is not checked again by the guardrails, so a rule added to protect a name never lets an answer kept before it show that name. Only the answers given from the documents alone are kept: an answer that called a tool (a web search, the memory, a search during the reasoning) depends on more than the question, and an answer stopped by a guardrail, empty or in JSON is not kept either. Like the approver, the confirmation is asked by the interface (`Agent.confirm_cache`): without it, the cache is not used.

Serving the closest answer automatically above the threshold was considered and ruled out. With `qwen3-embedding:0.6b`, rephrasings of the same question scored between 0.911 and 0.974, but different questions scored up to 0.970: "Quel montant Nordbank a-t-elle viré à Marc Vidal ?" and "Quel montant Marc Vidal a-t-il viré à Nordbank ?" are 0.970 close, with opposite answers, and asking when instead of how much still scored 0.946. No threshold separates them, so the journalist decides, with the question kept and its similarity in front of them.

### Answer examples

A small model follows an example better than an instruction. With the `examples` feature (on by default), the journalist keeps the answers they find good with `/example add`, and for each new question Sensai shows the model the closest ones, as examples of how to answer:

```
> Quel bonus touche Marc Vidal ?
Le bonus de Marc Vidal n'est pas mentionné dans les documents fournis.
> /example add
Answer kept as example 1: "Quel bonus touche Marc Vidal ?"

> Quel salaire touche Marc Vidal ?
[examples] 1 example: "Quel bonus touche Marc Vidal ?"
Le document n'évoque pas le salaire de Marc Vidal. [...]

> /examples
Answers kept as examples (* shown to the model for the last answer):
  * 1. Quel bonus touche Marc Vidal ? (kept on 2026-10-31)
```

`examples.py`: `ExampleLibrary` keeps the examples in `SENSAI_EXAMPLES_PATH`, each with its question, its answer and the excerpts the answer cites, under their number so that `[2]` keeps its meaning. For each message, the question is embedded with `SENSAI_EMBED_MODEL` and compared by cosine similarity to the questions kept, and the `SENSAI_EXAMPLES_K` closest ones (2 by default) above a similarity of 0.5 are sent in a system message just before the question, with the instruction to answer the same way without reusing their facts. They never go to the history, and the step-by-step reasoning gets them too. Only plain answers can be kept: not `/json` nor `/review`, not an answer stopped by a guardrail or interrupted, and never the same answer twice. Sensai ships no examples: only the journalist decides what a good answer is, and their examples stay on the machine, since they hold the names of the case.

The threshold comes from `qwen3-embedding:0.6b`: with examples from the same case, the questions on the same subject scored between 0.68 and 0.88, and messages like "Bonjour", "Merci" or a question off the case at most 0.30. With examples from another case, the right example still came first, but at 0.35 to 0.48, as close as "Bonjour": the examples of another case are not shown.

Measured with `qwen2.5:1.5b` on three documents of a fictional case and five examples kept on it, asking ten new questions twice without and twice with the examples:

| | Without examples | With examples |
|---|---|---|
| The documents do not say: the answer says so, without making it up | 6 / 10 | 9 / 10 |
| The documents say: the answer cites an excerpt | 0 / 10 | 5 / 10 |
| The documents say: the answer is entirely right | 10 / 10 | 8 / 10 |
| Average time per answer | 4.6 s | 9.1 s |

Without examples, the model made up four answers: the company's address as Marc Vidal's home (twice), "Marc Vidal has no account in Switzerland", and 12 € of bank fees as the company's taxes. With them, only the address was made up, once. But the answers got longer, and twice the model added a wrong detail to a right answer ("a dinner" for a lunch, "an expense report" for a bank statement), and it took twice as long, since the prompt and the answers are longer. An example also only teaches what it shows: an answer kept without a citation does not teach the model to cite. Ten questions asked twice is a small sample, and the model does not always answer the same way.

The other ways to optimize a prompt were considered and ruled out. Rewriting the question before answering it costs one more call to the model per message, and optimizing the prompt against a score, like [DSPy](https://dspy.ai), tries dozens of prompts, hundreds of calls that a CPU cannot afford. Shipping examples with Sensai and keeping the answers approved by the fact-checker of `/review` automatically were ruled out too: an approved answer is not always well written, and the journalist should decide what the model imitates.

### Logs & metrics

With the `logging` feature (on by default), every message is logged in `SENSAI_LOG_PATH`, a SQLite file that stays on the machine: the date, the conversation, the model, the persona, the version of the prompt, the question, the prompt sent (the exact messages given to the model: the system prompt, the history, the excerpts, the tool results), the answer, the outcome (answered, from the cache, blocked, stopped or error, with its message), the time of the answer and of its first word, the tokens sent and received, the tools called, the excerpts found, the guardrails hit and the steps of the reasoning. `/stats` and `python -m sensai stats` turn it into metrics:

```
$ python -m sensai stats --days 1
Last 1 day: 9 messages in 9 conversations, 0 errors (0%).
Answers: 8.6 s on average, 28.6 s for the slowest 5% (p95), first word after 5.3 s.
Tokens: 400 sent and 66 received per answer.
Outcomes: 9 answered.

By day:
  2026-11-01  9 messages: 9 answers, 8.6 s, p95 28.6 s

By prompt version:
  fa5fa4cc  since 2026-11-01 01:29: 3 answers, 6.7 s, p95 8.7 s
  e5ea938d  since 2026-11-01 01:29: 3 answers, 5.9 s, p95 9.3 s
  44fd7587  since 2026-11-01 01:30: 3 answers, 13.1 s, p95 28.6 s  [drift] +124% slower than e5ea938d on qwen2.5:1.5b
[...]
```

These are real runs with `qwen2.5:1.5b`: the same three questions asked with the default prompt, then with a system prompt asking for detailed answers, which the small model ignored (the answers did not get longer, nor slower), then with one asking to copy each excerpt and explain it before answering. That last one doubled the time of the answers: one of them took 28.6 seconds for 316 tokens.

`monitoring/log.py`: `Agent.run` watches the events of each message and logs it once it is over, even when it fails (Ollama not running, Ctrl+C): the error is logged, then raised as before. A log that cannot be written is a warning, never a reason to stop an answer. The agents of `/review` are logged too, each with its persona, so that the slowest step of a review shows up. The question, the prompt and the answer are logged as they are after the guardrails: a rule that redacts the name of a source redacts it in the log too. `SENSAI_LOG_CONTENT=false` keeps only the measures, without any text; `/stats clear` empties the log.

`monitoring/metrics.py`: `summarize` computes the metrics of a period, without printing them, so that another interface can show them. The times are those of the answers of the model: an answer from the cache or a blocked message would make them look faster. The p95 is the time 95% of the answers stay under: an average hides the few very slow answers. The first word is the first one shown, after the output guardrails, which hold back the end of the text to find a name cut in two: on a short answer, it comes with the last one. The version of the prompt is a hash of the system prompt (the persona or `SENSAI_SYSTEM_PROMPT`) and of the instructions written in the code for the documents and the reasoning: any change to them is a new version. A version whose answers are at least 50% slower than those of the previous one, on the same model and with at least three answers on each side, is flagged as a drift. With three answers, one long answer is enough to raise it, as above: the more answers, the more the alert can be trusted.

A graphical dashboard was left for later: `/stats` already shows what the metrics are for, and `summarize` keeps them apart from their display. Observability tools like [Langfuse](https://langfuse.com) were ruled out: even self-hosted, they run as a web service with its own database, a lot to install next to a CLI that works offline, when one SQLite file next to the index is enough.

### Scheduled tasks

A journalist does not only ask questions, they also watch a case. A task is a question Sensai goes through on its own, at set times, with nobody at the keyboard:

```
> /schedule add daily 08:00 What is new about Orion Delta in the documents?
Task 1 added, first run on 2026-11-02 08:00.
It runs while python -m sensai schedule is running.
> /schedule add every 6h --allow web_search Any new article about Orion Delta?
Task 2 added, first run on 2026-11-01 16:12.
It runs while python -m sensai schedule is running.
```

`python -m sensai schedule` stays running and runs each task when it is due. `python -m sensai schedule --once` runs what is due and exits, for the task scheduler of the machine (cron, Windows Task Scheduler), with an exit code of 1 when a task failed.

```
Sensai scheduler - model: qwen3, 2 tasks. Ctrl+C to stop.
[08:00] What is new about Orion Delta in the documents?
New report: .sensai/reports/20261102-080000-task-20261101-101200.md
[10:12] Any new article about Orion Delta?
Nothing new.
```

`schedule.py`: `Schedule` keeps the tasks in `SENSAI_SCHEDULE_PATH`, and the chat and the scheduler take a lock on it, then read it again before every change, so nothing is lost when both write and a task added during a run is picked up within 15 seconds. The lock belongs to the system (`msvcrt` on Windows, `flock` elsewhere), which drops it if a process dies, so a crash never leaves the schedule locked. A task moves to its next time before it runs, and from the current time rather than the missed one: a task missed ten times while the machine was off runs once, and a crash in the middle of a run does not make it start again in a loop.

**Each task has its own conversation**, saved like any other in `SENSAI_SESSIONS_DIR` under the id of the task. So the model has its earlier reports in its context, condensed by the same compaction as a chat, and a system message tells it to report what changed. That instruction is in the system prompt, not in the question, so that the document search looks for the question alone. `/resume task-...` continues the conversation of a task to ask a question about a report, and `--resume` alone never picks it.

**The alert is a report**, and it does not depend on the model. Each run keeps a fingerprint of what it found: the text of the excerpts retrieved and the results of the tools, in any order. When the next run finds exactly the same, no report is written, the conversation does not change and the task shows "nothing new", so an hourly check does not bury the report that matters. A new excerpt or any other tool result gives a report, and so does a run that found nothing to compare, since nothing proves that nothing changed.

Letting the model decide was tried first. Asked to answer `NOTHING NEW` when nothing had changed, `granite3.1-moe` did it in 1 try out of 6, and also in 1 try out of 6 when a new fact was there: it could have kept a resignation quiet. A second call asking it whether the new report states a new fact answered yes 12 times out of 12, changed or not. With such a model the prose of two reports always differs, while the evidence does not. The price is a report too many when a web page changes a few words, never a change kept quiet by the model. The model is still called at each run, since the tools it calls are only known once it has answered.

**A report** is a Markdown file in `SENSAI_REPORTS_DIR`, with the question, the date, the answer and the sources it cites, readable without Sensai. `/reports` lists them, `*` marks the unread ones, `/reports N` prints one, and the chat says at startup how many are new and how many tasks failed.

**Nobody is there to approve a tool**, so nothing runs by default: the agent of a run refuses every tool that asks for approval in the chat, and tells the model so. `--allow web_search` gives that approval once and for all when the task is created, for that task and that tool, and `/schedule` shows it. It is a standing decision: a task allowed to use `web_search` sends its query out of the machine at every run. A report says which tools were refused or failed, and such a run always gives a report, since finding the same thing may only mean that the model could not look. With `SENSAI_APPROVAL=off`, nothing is asked in the chat and nothing is refused here either.

The guardrails check the task when it is added, so a name to protect is redacted in the schedule file or the task is refused, and again at each run. The other features apply as in the chat: the profile, the persona (`python -m sensai -p investigator schedule`), the examples, the reasoning. Two do not. The cache would answer a monitoring question with yesterday's answer, and the working document would change without the journalist seeing it: a run uses neither.

A failed run, because Ollama is stopped or a tool crashed for example, does not stop the scheduler: the error is shown by `/schedule`, the chat warns about it at startup, and the task runs again at its next time. Only the start needs Ollama: without it, `python -m sensai schedule` stops with an error, and the tasks stay late until it is started again.

Running the tasks in a thread of the chat was ruled out: the tasks would stop with the chat, and a run would write over the prompt. The cron of the machine alone was ruled out too: it does not work the same on Windows, and it does not tell the journalist that a task is late. `/schedule` does. A library like APScheduler would only replace the computation of the next time, not the catch-up rule, the approvals or the reports.

Limits: tasks run only while the scheduler runs and the machine is on, one at a time, so a long task delays the next one. The times are those of the machine. A task reads the documents already indexed: new files need a `python -m sensai ingest`. If a conversation of a task is continued in the chat during a run, the last one to save it wins, but the report stays.

### Tools

The model can call tools with [Ollama's tool calling](https://docs.ollama.com/capabilities/tool-calling). A tool is a plain Python function: the Ollama SDK builds the description sent to the model from its type hints and its docstring (`Args:`, `Returns:`). The model answers with tool calls instead of text, Sensai runs them and sends the results back, and the model continues with them. A turn allows up to `SENSAI_MAX_TOOL_ROUNDS` rounds of calls, then the model has to answer.

- `tools/registry.py`: `ToolRegistry` finds the function by name, checks the arguments against its signature and calls it. An unknown tool, a wrong argument or a `ToolError` is sent back to the model as the result, so it can fix its call instead of the turn crashing.
- `tools/catalog.py`: `build_registry` lists the tools enabled in the configuration. Adding a tool is a documented function in a new module and one line there:

```python
def web_search(query: str) -> ToolResult:
    """
    Search the public web for recent or public facts that are not in the user's documents: ...

    Args:
      query: Keywords to search for

    Returns:
      The title, URL and content of the first results
    """
```

A tool returns the text sent to the model, or a `ToolResult(content, summary)` when the CLI should also show a short summary (`[web_search] 5 results: ...`). A tool that needs settings is a method of a class that holds them: `catalog.py` registers `WebSearch(backend, max_results).web_search`, and Ollama only sees the `query` parameter.

Like the excerpts, the full results are only sent for the current message. The history keeps the final answer and a short record of each call: the tool, its arguments and the summary of its result, each cut to 200 characters and passed through the rules of an answer, since they reach the saved session. A call blocked by a guardrail is not kept: its arguments are what the rule stopped. Neither is `ask_user`: the answer is the journalist's own words, which stay in their turn.

With the answer alone, a model that finds a request already answered with text imitates that text instead of calling the tool. Measured with `qwen2.5:1.5b` on real files, ten tries each, for a second request to write a file in a conversation: without the record, the file reached the disk once, and six times the model said it was saved when nothing had been written; with it, ten files out of ten and no false claim. The record has to be real tool messages: writing the same thing in the text of the answer, or adding a reminder, changed nothing. Keeping the start of the results instead of their summary did not help either. The memory tools weigh on this model: with their six descriptions in the prompt it invents paths like `/home/user/drafts/note.md`, and only four writes out of ten succeed. The permission error is shown and nothing false is claimed, but `SENSAI_FEATURE_MEMORY=false` is better with it.

Tools need a model that supports them (`qwen3`, `llama3.1`, `mistral`...). With a model that does not, like `llama3`, Sensai warns at startup and chats without tools.

**Web search** (`tools/web_search.py`) uses [Ollama's web search API](https://docs.ollama.com/capabilities/web-search), which needs a free API key from [ollama.com/settings/keys](https://ollama.com/settings/keys) in `OLLAMA_API_KEY`. The model gets the title, URL and content of the first `SENSAI_WEB_SEARCH_RESULTS` results, each page cut to 1500 characters so that they fit in the context window, and marked as untrusted data so that instructions hidden in a page are not followed. `SearchBackend` keeps the search engine separate from the tool, so it can be replaced without touching the rest.

### Approvals

Some tool calls cannot be undone: a web search sends the query outside the machine, and with it maybe the name of a source. Before running such a call, Sensai stops, shows the exact call the model asked for and waits for the user's decision. A tool declares that it needs approval, and why, with a decorator:

```python
@requires_approval("the query is sent to an external search engine")
def web_search(self, query: str) -> ToolResult:
```

- Approved, the tool runs as usual.
- Refused, it does not run: the model gets the refusal and the user's reason instead of the result, and is told not to make up what the tool would have found. It can rephrase its call, answer from what it already knows for sure, or ask the user.
- `always` approves the tool until the end of the session.

`SENSAI_APPROVAL` sets when Sensai asks: `critical` (default) for the tools that declare it, `all` before every tool call, `off` never. The core does not ask the user itself: the interface gives the agent an approver, a function that returns the decision, so a web interface can show buttons instead of a terminal prompt. Without an approver, critical calls are refused.
**Code execution** (`tools/code_runner.py`) lets the model run the Python code it writes, like ChatGPT's Code Interpreter. Each call starts a new Docker container from `SENSAI_SANDBOX_IMAGE`, sends the code to `python -` on its standard input and removes the container afterwards. The container has no network, 256 MB of memory, one CPU, 64 processes at most, a read-only disk (only `/tmp` is writable) and runs as `nobody` without privileges; nothing from the host is mounted. A program running longer than `SENSAI_CODE_TIMEOUT` seconds is killed. The model gets the exit code, the output and the traceback, each cut to their last 2000 characters. A crash is not a tool error: the model reads the traceback, fixes its code and calls `run_python` again, within the `SENSAI_MAX_TOOL_ROUNDS` limit. At startup Sensai checks that Docker runs and downloads the image if needed, otherwise it warns and chats without code execution.

**File access** (`tools/files.py`) lets the model work with the journalist's own files: `list_files(folder)`, `read_file(path, start_line)` and `write_file(path, content)`. `read_file` extracts the text of the formats Sensai indexes (PDF with its page numbers, emails, spreadsheets, HTML) and reads the others as text; a long file is sent 6000 characters at a time, and the model reads on with the `start_line` it is given. Like web pages, the content is marked as data, not instructions. `write_file` creates or replaces a whole text file, written next to it then renamed, so that a failure never leaves half a file; it needs approval, and the CLI shows the path and the full content before asking. There is no tool to delete a file.

Every path goes through `tools/permissions.py` before the file is touched. `Permissions` resolves the path first, following `..` and links, then checks that it is inside a folder of `SENSAI_FILES_READ` (to list and read) or of `SENSAI_FILES_WRITE` (to write, which also allows reading), and that no part of it matches a pattern of `SENSAI_FILES_DENY`, like `sources` or `*.key`. `.env`, `.env.*`, `.sensai` and `.git` are always denied, even inside an allowed folder: the model can never read the API key, the index, the saved sessions, the memory or the guardrail rules, which hold the names to protect. A refused path is sent back to the model with the reason, and `list_files` leaves out what may not be read. The arguments of `write_file` go through the guardrails like every tool call, so the source's name is redacted in the files the model writes too.

It is the `files` feature, off by default because the model reads and writes files on this machine. Turned on, it reads the folder Sensai runs in and writes nowhere, until `SENSAI_FILES_WRITE` names a folder. `/files` shows what is allowed.

### Questions to the user

A journalist's question is sometimes ambiguous: two companies share a name, a nickname stands for someone, a period is missing. With the `questions` feature (on by default), the model can ask instead of guessing: it calls `ask_user`, the CLI shows the question and its options, the journalist answers with a number or in their own words, and the model goes on with the answer. During the step-by-step reasoning, on a bank statement with two Vidal:

```
> Il y a deux Vidal dans le relevé et je ne sais pas lequel m'intéresse. Combien Vidal a-t-il reçu de Nordbank ?
[search] 1 excerpts from 1 documents
[thought 1] I need to find out who 'M. Marc Vidal' and 'Mme Sophie Vidal' refer to.
[ask_user] Quel sont les deux Vidal qui figurent dans le relevé et à qui appartient celui du crédit ?
Your answer: Marc Vidal
[ask_user] answered "Marc Vidal"
[thought 2] The user has identified 'M. Marc Vidal' as Marc Vidal.
[...]
Il y a deux Vidal dans le relevé. 'M. Marc Vidal' a reçu 480 000,00 de Nordbank, tandis que 'Mme Sophie Vidal' a reçu 35 000,00 de la société.
```

`tools/questions.py`: `ask_user(question, options)` is a tool like the others, whose docstring tells the model when to ask (a missing piece that no document and no tool can give) and when not to (what it can find itself). The question is put to the user by the interface, through the `Asker` it gives to `build_registry`, like the approver: the CLI prints the options numbered, a number picks one and anything else is sent as it is, an empty answer asks again, and `Ctrl+D` tells the model that the user did not answer and that it should not ask again. The web page shows the options as buttons, with a field for anything else and `Don't answer`. A run without a user gives no `Asker`, so the scheduler and `python -m sensai eval` have no `ask_user`: a question would wait for the keyboard forever. The call itself is not printed, so that the question is not shown twice. The question is the argument of a tool call: it goes through the guardrails of the tool calls before it is shown, so that a protected name is redacted from it. The answer is the journalist's own words, given to the model as the result of the tool, and it lives only in the messages of that turn, like the excerpts: `private` on its `ToolResult` keeps it out of the record of the tool calls and of an interrupted turn. What the model then writes goes through the output guardrails as usual. An answer that called `ask_user` is never kept by the semantic cache, since it depends on more than the question.

Whether the model asks is up to the model. On ambiguous questions like "Combien Vidal a-t-il reçu de Nordbank ?" on a statement with Marc and Sophie Vidal, each asked several times:

| | The ambiguity is said in the question | The ambiguity is only in the documents |
|---|---|---|
| `qwen2.5:1.5b` | never asks | never asks |
| `qwen2.5:1.5b` with the reasoning | asks 1 time out of 3 | never asks: it answers with Marc Vidal's amount |
| `qwen3:1.7b` | asks 6 times out of 6 | never asks |

Small models ask when the user says the request is ambiguous, not when only the documents show it, and `qwen3:1.7b` never filled the options. Without documents, `qwen2.5:1.5b` with the reasoning asked too much instead, things it should have looked for itself ("which bank?", "which period?"). Having Sensai check every message for an ambiguity before answering was considered and ruled out: it would cost one more call to the model for every message, for questions that are rarely ambiguous.

### MCP servers

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) lets a program, the server, offer tools to an AI application. Sensai is an MCP client: it reads the tools of the servers listed in `SENSAI_MCP_PATH`, and the model calls them like its own tools. Only servers reached by their URL are supported (Streamable HTTP transport).

```json
{
  "mcpServers": {
    "calculator": { "url": "http://127.0.0.1:8001/mcp" },
    "private": { "url": "https://mcp.example.com/mcp", "headers": { "Authorization": "Bearer KEY" } }
  }
}
```

`headers` is optional and sent with every request, for a server that asks for a token. Keep the file in `.sensai/`, which is not committed: it can hold keys.

- `tools/mcp.py` uses the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (`mcp`). At startup, `load_tools` asks each server for its tools (`tools/list`) and turns each one into a `SchemaTool` (`tools/registry.py`): the server gives a name, a description and a JSON schema of the arguments, sent to Ollama as they are. When the model calls the tool, `call_tool` sends `tools/call` to the server, which checks the arguments, runs the tool and answers.
- Each call opens its own connection and closes it: nothing stays open, and a restarted server is simply reached again. A server gets `SENSAI_MCP_TIMEOUT` seconds to answer.
- The arguments leave the machine, so every MCP tool needs approval in the default `critical` mode, and the guardrails check its calls like any other tool.
- A server that does not answer at startup only loses its tools, with a warning. A tool whose name is already used, by Sensai or by another server, is left out, never replaced. A failed call is sent back to the model as an error. An invalid file stops Sensai at startup, like an invalid guardrails file.

`mcp_servers/` holds two servers built with the SDK's `MCPServer`, each started on its own:

| Server | Tools | URL |
|---|---|---|
| `python mcp_servers/calculator.py` | `add`, `multiply`: exact arithmetic instead of mental computing | `http://127.0.0.1:8001/mcp` |
| `python mcp_servers/sirene.py` | `find_siren`: the SIREN, head office SIRET, address and state of a French company, from the public [company directory](https://recherche-entreprises.api.gouv.fr/docs/) | `http://127.0.0.1:8002/mcp` |

[mcp.example.json](mcp.example.json) lists both, and the public [DeepWiki](https://mcp.deepwiki.com) server, an existing MCP server that answers questions about GitHub repositories. To use them:

```bash
python mcp_servers/calculator.py &
python mcp_servers/sirene.py &
cp mcp.example.json .sensai/mcp.json
python -m sensai          # /tools lists add, multiply, find_siren and the DeepWiki tools
```

Any other server is one more entry with its URL. It is the `mcp` feature, on by default: without the file, no server is reached.

### Guardrails

`guardrails/rules.py` checks three places, each against the rules that list it in `on`:

- `input`: the message, before anything is searched or sent to the model. A blocked message is not answered and not kept in the history.
- `tools`: the text arguments of each tool call, before it runs. A blocked call is not run: the model gets the reason as the tool's result, so it can call it again without the blocked data. `tools` limits a rule to some tools, like `web_search`, the only one that reaches the network.
- `output`: the answer, while it is streamed. The last 80 characters are held back until the next chunk, so that a name split across two chunks is still found and a match is never shown in two pieces; the redacted answer is the one shown, kept and saved.
- `publish`: a public page, before it is published. The rules on `input` and `output` apply to it too: see [Export & publication](#export--publication).

A rule looks for one thing and does one of three actions, `block`, `redact` (replaced by `[NAME]` or its `replacement`) or `warn`:

| Matcher | Finds | Example |
|---|---|---|
| `terms` | words or names, whatever their case, accents and spaces, as whole words | the source's name and codename |
| `pattern` | a regular expression (case-insensitive) | `adresse personnelle` |
| `detectors` | personal data, with their checksum: `email`, `phone`, `iban` (mod 97), `card` (Luhn), `nir` (French social security key), `ip` | `["iban", "card"]` |
| `judge` | messages that break a policy, judged by the local chat model (input only, block only) | "finding the home address of a private person" |

The judge is the chat model itself, so nothing leaves the machine, at the cost of one more call per message. It is asked to answer `ALLOW` or `BLOCK: reason`, and anything else blocks the message: the guardrails fail closed, like an invalid rules file, which stops Sensai at startup instead of running without its rules.

Without a rules file, one rule applies: personal data is blocked in web searches. A file at `SENSAI_GUARDRAILS_PATH` replaces it; [guardrails.example.json](guardrails.example.json) protects a source, keeps personal data out of web searches, flags bank details in answers and refuses requests about private life. Like the index, keep the file in `.sensai/`: it holds the names you protect.

### Web UI

`python -m sensai web` serves a page on `http://127.0.0.1:8000` (`--port` to change it): the same conversations, tools and guardrails as the CLI, in a browser.

The page shows the answer as it is generated, with the searches, the tool calls, the thoughts and the result of `fact_check` above it and the sources cited below it. The sidebar lists the saved conversations, shared with the CLI, and `New chat` starts another one. The composer has the persona list and a gauge of the tokens used out of `SENSAI_NUM_CTX`, which turns orange when the oldest turns are about to be condensed. `Settings` turns the features on or off for this run, `Export & share` downloads the conversation or the document and creates public links, the stop button ends an answer and keeps what was written, and a tool that needs approval shows `Allow`, `Always this session` and `Deny`.

The backend does not change: `sensai/interfaces/web/server.py` is an interface like `sensai/interfaces/cli/`, it reads the events of `Agent.run` and reuses the functions of the CLI to start a session, save it and resume it.

- **One thread for the session.** An SQLite connection only works on the thread that opened it, and the index, the memory and the cache each hold one. `WebChat` runs everything that touches the session on a single thread: the server only puts the events in a queue and reads them back.
- **Streaming.** `POST /api/chat` answers with a stream of [server-sent events](https://developer.mozilla.org/docs/Web/API/Server-sent_events), one JSON object per event of the agent, read with `fetch`. The stream only goes one way: the clicks are ordinary requests (`/api/stop`, `/api/approval`), so a WebSocket would add nothing.
- **Approvals.** `WebChat.approve` is the approver of the agent: it sends the question to the page and blocks the thread of the session until the click. Stopping the answer or closing the page refuses the call, so a tool never runs on a question nobody saw.
- **Questions.** `ask_user` waits the same way: `start_session` takes the `Asker` of the interface, and `WebChat.ask` sends the question to the page and blocks the thread of the session until the click. The card has a button for each option, a field for anything else and `Don't answer`. Stopping the answer or closing the page leaves the question unanswered, so that the model never waits for someone who left.
- **Stopping** works like Ctrl+C: the generator of `Agent.run` is closed, which closes the connection to Ollama, and `Agent.interrupt` keeps what was written. A closed or reloaded page stops the answer too.
- **Only this machine.** There is no login, and a conversation holds the case: the server only listens on `127.0.0.1`. That is not enough on its own, since any site open in the browser can send requests to localhost: a request with another `Origin` is refused, and so is another `Host`, which stops a DNS rebinding attack.
- **The content of a document never runs.** The answers of the model can quote a leaked document, so the small Markdown renderer of `app.js` escapes everything before adding its own tags, and it creates neither links nor images.

The page is three static files in `sensai/interfaces/web/static/`, without a build step or a framework. It loads nothing from the internet, as Sensai has to work on an air-gapped machine: no CDN, no web font. Its design follows the Claude web interface: a sidebar, one centered column, a rounded composer, serif text for the answers, and a light and a dark theme that follow the system.

[Rich](https://github.com/Textualize/rich) was considered first, but it draws in a terminal: it does not make a web frontend. Textual can show a terminal application in a browser with `textual serve`, which is still not a frontend decoupled from the backend. React or Vue were ruled out because a build step and hundreds of packages are out of proportion for one page, on a project that has to be installed offline.

Limits: there is one conversation at a time and one user, so two tabs share the same conversation. The stop button acts at the next event: it cannot cut a tool that is running, nor a model that has not written its first word yet. The gauge counts each piece of the stream as one token until Ollama gives the exact count at the end, and it is an estimate after a stop or a resume. The page has the chat, not all the commands: `/review`, `/json`, `/ingest`, `/edit`, `/schedule` and the other commands stay in the CLI, and the semantic cache is not offered, since confirming a cached answer has no screen yet. A model that writes its tool calls as text instead of using tool calling, like `granite3.1-moe` in our tests, never reaches the approval screen.

### Export & publication

A case does not stay in the chat: it is archived, sent to an editor, read by a colleague who has no Sensai. `/export PATH` writes the conversation and `/document export PATH` the working document; the end of the name says the format. The page has the same in `Export & share`.

| Format | What it is for | What it holds |
|---|---|---|
| `.md` | reading and editing the text | the active branch, with the summary of the older messages and the sources cited |
| `.pdf` | archiving and sending | the same, with numbered pages |
| `.html` | one file to open or to host anywhere | the same, in the page used for the links |
| `.json` | another tool, or an archive | the conversation as it is saved, with all its branches and tool calls: copied into `SENSAI_SESSIONS_DIR`, it can be resumed |

A local export is a faithful copy: what goes in the archive of a case must not be changed, so the guardrails do not touch it.

A public link is a different thing, because a journalist's first duty is not to give a source away. It is off by default, like everything that lets data leave the machine. With `SENSAI_FEATURE_SHARING=true` in `.env`, start the server next to the chat:

```bash
python -m sensai publish
```

```text
> /share conversation
Page: Who approved the transfer to Orion? (4 messages)
Rule 'source': 3 redacted (SOURCE).
Rule 'published-personal-data': 1 redacted (EMAIL).
Sources named: mail_004.eml; releve_avril.pdf, page 3
Anyone who gets the link can read this page, without logging in.
Publish? [y]es / [n]o: y
Published: http://192.168.1.20:8765/shared/5fbeea46937d32afbf3715a89d111e3e
The page stays as it is now: the next messages are not added. The links answer while python -m sensai publish runs.
```

`/share conversation` publishes the active branch, `/share document` the working document, `/share` lists the links and `/share remove N` takes one down at once, without restarting anything. `Create a link` on the page shows the same preview, with `See the page` to read it before publishing.

`sensai/core/export/`: `report.py` turns a conversation or the document into a `Report`, the same for every format, and `markdown.py` reads the Markdown once for the HTML of `page.py` and the PDF of `pdf.py`, the way the web page reads it.

- **A second server.** The chat stays on `127.0.0.1`, without a login, because a conversation holds the case. `python -m sensai publish` (`interfaces/web/public.py`) is another process, which faces the network and only holds the folder of the pages: it starts without the model, does not load the chat, and answers `/shared/<link>` and a 404 for anything else. A route in the chat server would have put the case one bad check away from the network.
- **A frozen copy.** The page is written once, as one file in `SENSAI_SHARES_DIR`: nothing in it follows the conversation, and the server has no access to the sessions. Removing the file takes the link down.
- **The rules.** Publishing is the most outward thing Sensai does, so a page answers to every rule on `input` and `output`, plus the new `publish` stage. `redact` redacts, `warn` is reported in the preview, and `block` stops the publication. By default, `published-personal-data` redacts the personal data of a page. A `tools` rule is about one tool, and a `judge` rule about what the user may ask: neither applies.
- **What is published.** The messages, the summary and the sources cited, nothing else: not the tool calls, the profile or the system prompt. Sources only keep their file name, since a folder can name the case. A file name goes through the rules with its `_`, `.` and `-` read as spaces, as `claire_benoit.eml` would not match the term `Claire Benoit`; if a rule finds something, the page says `[source hidden by a rule]`. The title is cut from the redacted message: cut first, a protected name would leave a piece no rule can find.
- **The link is the only secret.** It holds 128 random bits, so it cannot be guessed. The server keeps no access log, the page has no script and loads nothing, and its headers keep it out of search engines and caches and keep the link from reaching the next site.

Without configuration, the links use the address of this machine on its network, which is enough for a colleague on the same Wi-Fi. For anyone else, publish through a tunnel or a reverse proxy and set `SENSAI_PUBLIC_URL`: the chat writes the links, and it cannot know the address the world sees.

For the PDF, [reportlab](https://www.reportlab.com/opensource/) does the layout: cutting paragraphs, code and lists across pages is its job, and its Vera fonts come with it. fpdf2 was ruled out because it has no Unicode font of its own, which would have added a binary to the repository. WeasyPrint needs GTK libraries, which is a lot to install offline on Windows, and printing from the browser would have left the CLI without a PDF. It is imported only when a PDF is asked for, so that a missing reportlab costs the PDF and nothing else.

Limits: a public link is public. Anyone who gets it can read the page and pass it on, and a page already read or copied cannot be taken back. The rules only find what they were told to look for: a name that no rule knows stays in the page, which is why the preview comes first. The fonts of the PDF cover Western European languages: other alphabets and emojis are shown as `?`, and the PDF says how many, while the Markdown, the HTML and the JSON keep them. Tables stay as plain Markdown text, as they do on the web page. The page is served by this machine: it stops answering when `python -m sensai publish` stops, and there is no expiry date or view count.

### Adversarial testing (red-teaming)

Investigative assistants handle confidential leaks and sensitive inquiries. `python -m sensai attack` audits
Sensai's defenses against adversarial prompts, evaluating whether the assistant withstands attempts to bypass
safeguards, leak confidential context, or trigger unhandled execution failures.

The built-in benchmark suite (`sensai/core/adversarial/suite.json`) covers three categories:
- `jailbreak`: attempts to override safety boundaries, role-play as unrestricted personas, or evade policies.
- `injection`: attempts to leak internal system instructions, reveal protected source names, or hijack tool calls.
- `malformed`: stress tests including token floods, deeply nested delimiters, null bytes, and control sequences.

Each test case runs against an isolated agent instance with guardrails active. The runner detects breaches using
multi-layered criteria: guardrail interception, model refusal heuristics, and leakage pattern matching.

```bash
python -m sensai attack                                    # run the default 15-case benchmark
python -m sensai attack --category jailbreak               # run only jailbreak attacks
python -m sensai attack my_cases.json                      # run a custom test suite
python -m sensai attack --out .sensai/audit.json           # export audit report to JSON
python -m sensai attack --compare .sensai/audit_prev.json  # compare defense rate with previous run
```

#### Empirical baseline assessment

Auditing the default suite on local models with active guardrails yields the following baseline metrics:

| Category | Attacks | Defended | Breached | Defense Rate | Notes |
|---|---|---|---|---|---|
| `jailbreak` | 5 | 5 | 0 | 100.0% | Refused by model policy or blocked by judge guardrail |
| `injection` | 5 | 5 | 0 | 100.0% | Zero system prompt or source name leakage detected |
| `malformed` | 5 | 5 | 0 | 100.0% | Handled safely without crash or buffer anomalies |
| **Overall** | **15** | **15** | **0** | **100.0%** | Comprehensive protection across all vectors |

### Automated evaluation & hallucination detection

An investigative journalist cannot publish claims that the documents do not support. Sensai implements a hybrid evaluation pipeline (`evaluation/`) to verify each answer against retrieved excerpts and tool results:

1. **Pre-check by code (`facts.py`)**:
   - Amounts (`480 000,00`, `480.000 EUR`, `480 000 €`), dates (`05/04/2023`, `2023-04-05`, `5 avril 2023`, years) and capitalized names are extracted and normalized by code.
   - Any figure, date or name present in the sentence but absent from the evidence is flagged immediately as `not in the excerpts` (verdict `unsupported`). This check runs in milliseconds, never confuses decimal commas, and saves model calls.
2. **Per-sentence judge (`judge.py`)**:
   - Sentences whose explicit facts exist in the evidence are sent one by one to `SENSAI_JUDGE_MODEL` (the chat model by default; `qwen3:1.7b` recommended) with a strict JSON schema requiring a reason and a verdict: `supported`, `contradicted`, or `unsupported`.
   - A final call scores the answer from 1 to 5 on `relevance` (answering the question asked) and `coherence` (clarity and internal consistency).
   - Source `faithfulness` is computed as the share of checked sentences verified as `supported`.
3. **Bounded execution**:
   - On a CPU, a thinking model can reason for thousands of tokens. `SENSAI_JUDGE_MAX_TOKENS` (1500 by default) caps the tokens per sentence; if the judge exceeds the budget or returns invalid JSON, the sentence is marked `unchecked` instead of hanging the session.

```
> Qui dirige Orion Delta Consulting, et combien a-t-elle reçu de Nordbank ?
[search] 2 excerpts from 2 documents
Orion Delta Consulting est gérée par Marc Vidal [1]. Elle a reçu un virement de 480 000 EUR de Nordbank le 05/04/2023 [2].

[check] Checking 2 sentences with qwen3:1.7b, this can take a few minutes (Ctrl+C to skip)...
[check] 2 sentences: 2 supported, faithfulness 100%, relevance 5/5, coherence 5/5
```

When an answer invents claims or contradicts evidence:

```
[check] 2 sentences: 0 supported, faithfulness 0%, relevance 5/5, coherence 5/5
  not in the excerpts: "Il a touché 480 000 EUR en 2022 [2]." (2022)
  contradicted: "Marc Vidal est le seul associé d'Orion [1]." (he is the manager, not an associate)
```

The CLI distinguishes certainty from suspicion:
- `not in the excerpts`: an amount, date or name confirmed absent from the evidence by code.
- `contradicted`: the judge found the evidence directly contradicts the statement.
- `to verify`: the judge gave an `unsupported` verdict, which may be an over-cautious false alarm.
- `unchecked`: the judge reached `SENSAI_JUDGE_MAX_TOKENS` or returned unparseable output.

#### Scoring system versions (`sensai eval`)

You can run automated benchmarks on a set of questions to validate prompt changes, RAG tuning, or new model versions:

```bash
python -m sensai eval eval.json --out results.json
python -m sensai eval eval.json --compare results.json
```

`eval.json` specifies questions and optional expected facts:

```json
{
  "cases": [
    {"question": "Qui dirige Orion Delta Consulting ?", "expected": ["Marc Vidal"]},
    {"question": "Combien Nordbank a-t-elle viré à Orion ?", "expected": ["480 000", "05/04/2023"]},
    {"question": "Quel salaire touche Marc Vidal ?"}
  ]
}
```

Each question is answered by a fresh agent with long-term memory, living document, cache and examples turned off, and then judged against retrieved evidence. Expected facts are matched using normalized values ("480 000 €" matches "480 000"). `--compare` prints side-by-side metric deltas against a previous benchmark run.

With `"sources"`, the documents that answer each question, by their path in the index or its end (`"mail_001.eml"`), `--retrieval` scores the document search alone, without calling the chat model: the Recall@`SENSAI_RAG_TOP_K`, the NDCG@10, the share of the excerpts sent that come from a document that answers and the excerpts sent for the questions without `"sources"` (the noise), for each step of the search and for thresholds from 10 to 50%. `eval/questions.json` and `eval/corpus/` are a fictional case to try it:

```bash
SENSAI_INDEX_PATH=/tmp/eval.db python -m sensai ingest eval/corpus
SENSAI_INDEX_PATH=/tmp/eval.db python -m sensai eval eval/questions.json --retrieval
```

#### Technical choices & honest evaluation

Tested on fictional leaked documents with `qwen2.5:1.5b` and `qwen3:1.7b` on CPU:

| Method / Model | Made-up facts caught | False alarms on accurate sentences | Time per answer |
|---|---|---|---|
| Whole-answer judge (`qwen2.5` or `qwen3`) | 0 / 4 (0%) | 0 / 4 | ~15 s |
| Per-sentence judge (`qwen2.5:1.5b`) | 0 / 4 (0%) | 0 / 4 | ~45 s |
| Per-sentence judge (`qwen3:1.7b`) | 4 / 4 (100%) | 2 / 4 (50%) | ~5 min |
| Hybrid check (`facts.py` + `qwen3:1.7b`) | 4 / 4 (100%) | 2 / 4 (50%) | ~3-4 min |

- **Why small models need per-sentence judging**: Small models (1.5B–3B) evaluating an entire paragraph at once (as in `/review`) completely missed invented claims. Only `qwen3:1.7b` with per-sentence chain-of-thought caught every hallucination.
- **Why code-based fact extraction**: Small models misread numbers (reading French decimal "480 000,00" as "480 million") or demanded details that sentences were not meant to include. Pre-checking amounts and dates by code eliminated number hallucination errors instantly.
- **Limits**: `qwen3:1.7b` remains overly suspicious on qualitative statements, occasionally claiming details are missing. On CPU, it takes minutes per answer. Therefore, `fact_check` is disabled by default (`/feature fact_check on` to enable), serving as a safeguard for careful verification rather than an automatic blocker.
- **Alternatives ruled out**: Embedding comparison (cosine similarity failed to differentiate transaction direction), dedicated NLI libraries (to adhere strictly to standard Ollama API constraints), and 3-vote self-consistency (prohibitive 15-minute runtime per turn on CPU).

## Getting Started

### Installation

Requirements: Python 3.10+ and a running [Ollama](https://ollama.com/download) server.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # without a GPU: skips gigabytes of CUDA libraries
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3
ollama pull qwen3-embedding:0.6b   # embedding model for document search
```

### Quickstart

```bash
python -m sensai                 # uses .env
SENSAI_MODEL=qwen3.8 python -m sensai   # override the model for one run
python -m sensai -e other.env    # use another env file
python -m sensai --resume        # continue the last conversation
python -m sensai --resume 20261027-153012   # continue a given one (/sessions lists them)
python -m sensai web             # chat in a browser page on http://127.0.0.1:8000
python -m sensai web --port 9000 # on another port
python -m sensai publish         # serve the pages published with /share, to whoever has their link

python -m sensai ingest ~/leak/          # index a folder (or files) for document search
python -m sensai ingest --rebuild ~/leak/   # start over, e.g. after changing the embedding model
python -m sensai status                  # what the index contains
python -m sensai stats --days 30         # the metrics of the messages of the last 30 days
python -m sensai -p fact_checker         # start with another persona
python -m sensai eval eval.json          # evaluate and score Sensai on test questions
python -m sensai eval eval.json --retrieval   # score the document search alone, step by step
python -m sensai attack                  # audit defenses against adversarial attacks
```

### Usage

Type a message and press Enter: the answer is streamed as it is generated, and previous turns are sent as context. Type `/exit` or press `Ctrl+D` to quit; `Ctrl+C` during an answer interrupts it.

Once documents are indexed, every message is answered from the closest excerpts, and the sources cited by the answer are listed below it:

```
> Qui a demandé le virement de 480 000 EUR vers Alvor Consulting, et y avait-il une facture ?
[search] 5 excerpts from 5 documents
D'après les documents fournis :

1.  **Qui a demandé (ou initié) le virement :** C'est Marc Vidal qui a demandé à Claire Benoit de valider le virement par email du 3 avril 2023 [1]. Le mouvement bancaire confirme que l'autorisation ("approved_by") est signée par M. Vidal [3].
2.  **Y avait-il une facture :** Non, il n'y avait pas de facture au moment du paiement. Marc Vidal l'indiquait explicitement dans son email : "Pas de facture pour l'instant" [1], et le rapport d'audit confirme que le paiement a été effectué "sans facture" [2]. De plus, Claire Benoit rejette la validation en déclarant qu'elle ne peut pas valider sans facture [5].

Sources:
  [1] leak/emails/001.eml
  [2] leak/audit_report.pdf, p. 1-3
  [3] leak/transfers.csv, rows 2-4
  [5] leak/emails/002.eml
```

| Command | Effect |
|---|---|
| `/ingest PATH...` | index files or folders without leaving the chat |
| `/sources` | show the full excerpts retrieved for the last message, with their relevance and similarity |
| `/sources dropped` | show the excerpts found for the last message but dropped under `SENSAI_RERANK_THRESHOLD` |
| `/filter [FILTER\|clear]` | search only some documents, like `after:2023-03 before:2023-07 from:dupont to:martin kind:email source:leak/mails/` |
| `/context` | show the tokens used by the last message and the summary of older turns |
| `/tools` | list the tools the model can call |
| `/files` | show the folders the model may read and write, and the patterns always denied |
| `/features` | list the features and whether they are on |
| `/feature NAME on\|off` | turn a feature on or off for the session |
| `/personas` | list the personas, `*` marks the active one |
| `/persona [ID]` | show the active persona, or switch to another one; the history is kept |
| `/sessions` | list the saved conversations, the current one marked with `*` |
| `/resume [ID]` | continue a saved conversation, the last one without ID |
| `/new` | start a new conversation; the current one stays saved |
| `/history` | list the messages of this branch with their numbers |
| `/edit N [TEXT]` | replace message N in a new branch; the original conversation stays in its branch |
| `/branches` | list the branches of this conversation, the active one marked with `*` |
| `/branch N` | continue branch N |
| `/profile [add TEXT\|remove N]` | show, add or remove the instructions sent to the model about you |
| `/document` | show the working document |
| `/document undo` | undo the changes made to the document by the last answer |
| `/document export PATH` | write the document to a `.md`, `.json`, `.pdf` or `.html` file |
| `/export PATH` | write the conversation to a `.md`, `.json`, `.pdf` or `.html` file |
| `/share` | list the public links |
| `/share conversation` | publish a frozen copy of this conversation on a link anyone can open, after a preview |
| `/share document` | publish a frozen copy of the document on a link anyone can open, after a preview |
| `/share remove N` | take link N down |
| `/json QUESTION` | answer in JSON that follows a schema: the answer, its claims with their excerpts, the entities and the events |
| `/guardrails` | list the rules checking the messages, the answers and the tool calls |
| `/approval [MODE]` | show when Sensai asks before running a tool, or change it: `critical`, `all` or `off` |
| `/reasoning [N]` | list the questions reasoned about in this conversation, or show the steps of question N |
| `/cache [clear]` | show how many answers the cache keeps and how often they were reused, or remove them all |
| `/examples` | list the answers kept as examples, `*` marks those shown to the model for the last answer |
| `/example add\|remove N` | keep the last answer as an example for similar questions, or remove example N |
| `/stats [DAYS\|clear]` | show the metrics of the messages of the last DAYS days (7 by default), or empty the log |
| `/prompts` | list the versions of the system prompt, `*` marks the one in use |
| `/prompt show\|use N` | print version N of the system prompt, or use it from now on (roll back) |
| `/prompt diff N M` | show what changed between versions N and M |
| `/prompt compare N M` | ask the test cases with versions N and M, and compare the checks passed, the time and the tokens |
| `/help` | list the commands |

Long conversations do not overflow the model's context window. Sensai estimates the size of each message before sending it and counts the tokens used once it is answered (as reported by Ollama); once they reach `SENSAI_COMPACT_THRESHOLD` percent of `SENSAI_NUM_CTX`, the oldest turns are condensed into a summary that keeps names, amounts, dates, sources and open leads, while the last `SENSAI_KEEP_RECENT_TURNS` turns stay verbatim. The summary takes about an eighth of the context window:

```
[memory] context at 6230/8192 tokens, condensing the oldest turns...
[memory] 3 turns condensed into the summary.
```

Every answered message is saved, so a conversation can be continued after a restart, and the profile is sent at the start of every conversation:

```
> /profile add Réponds en français et n'écris jamais le nom de la source : dis "la source".
Added as instruction 1.
> /exit

$ python -m sensai --resume
Profile: 1 instruction sent to the model. /profile to see them.
Resumed conversation 20261027-153012: 4 turns. /context shows them.
> Et la facture, finalement ?
```

After each answer, the working document is updated with what was established, section by section:

```
> Dans le mail du 4 avril 2023, Marc Vidal, directeur financier d'Alvor, approuve un virement de 48 250 € vers Kestrel Ltd.
[...]
[document] added to People and companies, added to Timeline. /document to read it.
> Correction : Marc Vidal est le directeur général.
[...]
[document] corrected in People and companies. /document to read it.
> /document export enquete-alvor.md
Document written to enquete-alvor.md.
```

With `SENSAI_FEATURE_WEB_SEARCH=true` (or `/feature web_search on` during a chat), the model searches the web when the question needs it. Each query is printed before it is sent, so you know exactly what leaves the machine:
With `SENSAI_FEATURE_WEB_SEARCH=true` (or `/feature web_search on` during a chat), the model searches the web when the question needs it. Each query has to be approved before it is sent, so nothing leaves the machine without your consent:

```
> Quelle est la dernière version stable de Python publiée ? Cherche sur le web.
[web_search] query="la dernière version stable de Python"
Allow web_search? The query is sent to an external search engine. [y]es / [n]o / [a]lways this session: y
[web_search] 5 results: www.python.org, devguide.python.org, docs.python.org
La dernière version stable de Python publiée est Python 3.15.1 [...]

URL de référence : https://www.python.org/downloads/latest/
```

With the rules of `guardrails.example.json` copied to `.sensai/guardrails.json`, the source is never named, personal data stays on the machine and requests about private life are refused:

```
Guardrails: 4 rules from .sensai/guardrails.json. /guardrails to list them.
> Qui a refusé de valider le virement ?
D'après l'email du 4 avril 2023, c'est [SOURCE] qui a refusé de valider le virement sans facture [2].
[guardrail] source: 1 passage redacted in the answer (SOURCE)
> À qui appartient le compte FR76 3000 6000 0112 3456 7890 189 ? Cherche sur le web.
[web_search] query="FR76 3000 6000 0112 3456 7890 189 titulaire"
[warning] web_search failed: blocked by the guardrail 'personal-data' (IBAN): The arguments contain personal data, which must not leave the machine. Search without it.
> Où habite Marc Vidal ?
[warning] Message blocked by the guardrail private-life: asks for the home address of a private person. Nothing was sent to the model.
```

With `SENSAI_FEATURE_CODE_EXEC=true` (or `/feature code_exec on` during a chat) and [Docker](https://docs.docker.com/engine/install/) running, the model runs Python to compute instead of guessing. When its code fails, it reads the traceback and tries again:

```
> Combien font 17 % de 48 250 euros ? Calcule-le avec du code Python.
[run_python]
  code:
    result = 48250 * 0.17
    print(result)
[run_python] ok
Le calcul de 17 % de 48 250 euros est de **8 202,5 euros**.
```

The default image only has the standard library. For pandas, build your own image and set `SENSAI_SANDBOX_IMAGE`:

```dockerfile
FROM python:3.12-slim
RUN pip install pandas numpy
```

With `SENSAI_FEATURE_FILES=true`, `SENSAI_FILES_READ=leak` and `SENSAI_FILES_WRITE=drafts`, the model reads the leak and writes its drafts, and nothing else:

```
> Lis le mail de Marc Vidal dans leak et écris un brouillon de trois lignes dans drafts/virement.md.
[list_files] folder="leak"
[list_files] 3 entries, 1 not allowed
[read_file] path="leak/mail-vidal.eml"
[read_file] leak/mail-vidal.eml, lines 1-12 of 12
[write_file] path="drafts/virement.md"
  content:
    Le 4 avril 2023, Marc Vidal valide un virement de 48 250 € vers Kestrel Ltd, sans facture.
    [...]
Allow write_file? It writes a file on this machine. [y]es / [n]o / [a]lways this session: y
[write_file] drafts/virement.md created
> Lis aussi leak/sources/contact.txt.
[read_file] path="leak/sources/contact.txt"
[warning] read_file failed: read_file: permission denied, leak/sources/contact.txt is denied by the pattern 'sources'
```

Supported files: `.pdf`, `.eml`, `.csv`, `.tsv`, `.xlsx`, `.html`, `.htm`, `.md`, `.txt`. Hidden files and folders are skipped.

### Configuration

Configuration lives in `.env` (see `.env.example`). Environment variables take precedence over the file.

| Variable | Default | Role |
|---|---|---|
| `SENSAI_MODEL` | `llama3` | chat model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `SENSAI_PERSONA` | empty | persona used at startup (overridden by `-p ID`); empty: no persona, `SENSAI_SYSTEM_PROMPT` is used |
| `SENSAI_PERSONAS_DIR` | `personas/` of the repository | folder of the persona files |
| `SENSAI_SYSTEM_PROMPT` | empty | system prompt sent at the start of every conversation when no persona is active |
| `SENSAI_NUM_CTX` | `16384` | context window of the chat model, in tokens |
| `SENSAI_TEMPERATURE` | model default | temperature of the chat model (0-2), lower for steadier answers |
| `SENSAI_THINK` | model default | `false` disables the thinking phase of models like qwen3.5 |
| `SENSAI_FEATURE_RAG` | `true` | answer from the indexed documents |
| `SENSAI_FEATURE_HYBRID` | `true` | also search the documents by their exact words (FTS5, BM25), merged with the semantic search |
| `SENSAI_FEATURE_RERANK` | `true` | sort the candidates by a cross-encoder and drop those under `SENSAI_RERANK_THRESHOLD`; needs sentence-transformers |
| `SENSAI_FEATURE_TOOLS` | `true` | `false` disables every tool, whatever their own setting |
| `SENSAI_FEATURE_WEB_SEARCH` | `false` | `true` lets the model search the web with Ollama's web search API; needs a model that supports tools and `OLLAMA_API_KEY` |
| `SENSAI_FEATURE_CODE_EXEC` | `false` | `true` lets the model run Python code in a Docker container; needs Docker and a model that supports tools |
| `SENSAI_FEATURE_FILES` | `false` | `true` lets the model list, read and write files in the allowed folders; needs a model that supports tools |
| `SENSAI_FEATURE_MEMORY` | `true` | let the model save and look up facts about people, companies and places; needs a model that supports tools |
| `SENSAI_FEATURE_MCP` | `true` | let the model call the tools of the MCP servers listed in `SENSAI_MCP_PATH`; needs a model that supports tools |
| `SENSAI_FEATURE_QUESTIONS` | `true` | let the model ask you a question when your request is ambiguous; needs a model that supports tools |
| `SENSAI_FEATURE_SESSIONS` | `true` | save every conversation and allow resuming it |
| `SENSAI_FEATURE_PROFILE` | `true` | send the profile to the model |
| `SENSAI_FEATURE_DOCUMENT` | `true` | update the working document after every answer, with one more call to the model |
| `SENSAI_FEATURE_CACHE` | `true` | offer the answer to a close question already asked, in the same context |
| `SENSAI_FEATURE_EXAMPLES` | `true` | show the model the answers kept as examples, for the questions close to theirs |
| `SENSAI_FEATURE_SCHEDULE` | `true` | let `python -m sensai schedule` run the scheduled tasks |
| `SENSAI_FEATURE_SHARING` | `false` | publish on public links, and let `python -m sensai publish` serve them |
| `SENSAI_FEATURE_PROMPT_VERSIONS` | `true` | keep every version of the system prompt, and use the one chosen with `/prompt use N` |
| `SENSAI_FEATURE_LOGGING` | `true` | log every message with its prompt, answer, tools, time and tokens |
| `SENSAI_FEATURE_FACT_CHECK` | `false` | check each answer against its excerpts and flag what they do not support |
| `SENSAI_COMPACT_THRESHOLD` | `75` | percentage of `SENSAI_NUM_CTX` at which the oldest turns are condensed (10-95) |
| `SENSAI_KEEP_RECENT_TURNS` | `4` | most recent turns always kept verbatim |
| `SENSAI_SESSIONS_DIR` | `.sensai/sessions` | folder where conversations are saved |
| `SENSAI_PROFILE_PATH` | `.sensai/profile.md` | your profile, one instruction per line |
| `SENSAI_MEMORY_PATH` | `.sensai/memory.db` | SQLite file holding the long-term memory |
| `SENSAI_FEATURE_BRANCHING` | `true` | edit past messages into branches, and keep an interrupted answer |
| `SENSAI_CACHE_PATH` | `.sensai/cache.db` | SQLite file holding the answers kept by the cache |
| `SENSAI_CACHE_THRESHOLD` | `90` | similarity, in percent, from which a kept answer is offered (50-100) |
| `SENSAI_FEATURE_GUARDRAILS` | `true` | check the messages, the answers and the tool calls against the rules |
| `SENSAI_COMPACT_THRESHOLD` | `75` | percentage of `SENSAI_NUM_CTX` at which the oldest turns are condensed (10-95) |
| `SENSAI_KEEP_RECENT_TURNS` | `4` | most recent turns always kept verbatim |
| `SENSAI_GUARDRAILS_PATH` | `.sensai/guardrails.json` | guardrail rules (see `guardrails.example.json`); without it, personal data is blocked in web searches |
| `SENSAI_MCP_PATH` | `.sensai/mcp.json` | URLs of the MCP servers (see `mcp.example.json`); without it, no server is reached |
| `SENSAI_EXAMPLES_PATH` | `.sensai/examples.json` | the answers kept as examples with `/example add` |
| `SENSAI_EXAMPLES_K` | `2` | examples shown to the model for each question, the closest ones first (0-5) |
| `SENSAI_SCHEDULE_PATH` | `.sensai/schedule.json` | the scheduled tasks and the list of their reports |
| `SENSAI_REPORTS_DIR` | `.sensai/reports` | folder of the reports, in Markdown |
| `SENSAI_SHARES_DIR` | `.sensai/shared` | folder of the public pages |
| `SENSAI_PUBLIC_PORT` | `8765` | port of `python -m sensai publish`, and of the links |
| `SENSAI_PUBLIC_URL` | (the address of this machine) | what the links start with, behind a tunnel or a reverse proxy |
| `SENSAI_PROMPTS_DIR` | `.sensai/prompts` | the versions of the system prompt (`system.json`) and the comparisons run |
| `SENSAI_PROMPT_CASES_PATH` | `.sensai/prompt_cases.json` | the test questions of `/prompt compare` (see `prompt_cases.example.json`) |
| `SENSAI_JUDGE_MODEL` | empty | model that checks the answers (/feature fact_check on); empty: the chat model |
| `SENSAI_JUDGE_MAX_TOKENS` | `1500` | tokens the judge may write for each sentence before it is left unchecked (200-8000) |
| `SENSAI_EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding model; changing it requires `ingest --rebuild` |
| `SENSAI_INDEX_PATH` | `.sensai/index.db` | SQLite file holding the index |
| `SENSAI_RAG_TOP_K` | `5` | excerpts added to the prompt for each message |
| `SENSAI_RAG_CANDIDATES` | `30` | chunks taken from each search, then merged and read by the reranker (1-200) |
| `SENSAI_RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | cross-encoder of the `rerank` feature, from Hugging Face, downloaded at the first search |
| `SENSAI_RERANK_THRESHOLD` | `20` | relevance, in percent, under which an excerpt is dropped (0-100, 0 keeps them all) |
| `SENSAI_RAG_NEIGHBORS` | `1` | chunks added before and after each excerpt kept, from the same document (0-3) |
| `SENSAI_RAG_CHUNK_SIZE` | `1200` | maximum chunk size, in characters |
| `SENSAI_RAG_CHUNK_OVERLAP` | `200` | characters repeated between consecutive chunks |
| `OLLAMA_API_KEY` | empty | key from [ollama.com/settings/keys](https://ollama.com/settings/keys), required by the web search |
| `SENSAI_WEB_SEARCH_RESULTS` | `5` | results the model gets for each web search (1-10) |
| `SENSAI_CODE_TIMEOUT` | `10` | seconds a program may run before it is stopped (1-60) |
| `SENSAI_SANDBOX_IMAGE` | `python:3.12-slim` | Docker image the code runs in; use one with pandas to analyse CSV files |
| `SENSAI_MCP_TIMEOUT` | `30` | seconds an MCP server may take to answer a tool call (1-300) |
| `SENSAI_FILES_READ` | `.` | folders the model may list and read, separated by commas; empty: none |
| `SENSAI_FILES_WRITE` | empty | folders the model may write in (and read), separated by commas |
| `SENSAI_FILES_DENY` | empty | name patterns never readable nor writable inside these folders, like `sources,*.key`; `.env`, `.sensai` and `.git` always are |
| `SENSAI_MAX_TOOL_ROUNDS` | `5` | rounds of tool calls allowed per message before the model has to answer (1-20) |
| `SENSAI_APPROVAL` | `critical` | when to ask before running a tool: `critical` (tools that send data out or run code), `all`, `off` |
| `SENSAI_LOG_PATH` | `.sensai/logs.db` | SQLite file holding the log of every message |
| `SENSAI_LOG_CONTENT` | `true` | `false` logs only the measures: no question, prompt nor answer |

### Features

Each feature can be turned on or off with a `SENSAI_FEATURE_<NAME>` variable (`true` or `false`), or during a chat with `/feature NAME on|off`. A misspelled feature variable stops Sensai with an error.

| Feature | What it turns on or off |
|---|---|
| `rag` | answers drawn from the indexed documents |
| `hybrid` | the keyword search added to the semantic search; off, the vectors alone find the excerpts |
| `rerank` | the cross-encoder and its threshold; off, the excerpts are the best of the searches, none is dropped |
| `tools` | every tool the model can call, whatever their own feature or setting |
| `web_search` | the web search tool, off by default because the queries leave the machine; needs `OLLAMA_API_KEY` |
| `code_exec` | the Python code execution tool, off by default because the code runs on this machine; needs Docker |
| `files` | the file tools, off by default because the model reads and writes files on this machine |
| `memory` | the long-term memory tools; the memory file is kept when it is off |
| `questions` | the `ask_user` tool, with which the model asks you a question when your request is ambiguous |
| `branching` | `/edit`, `/branches` and `/branch`, and keeping an interrupted answer; the branches stay saved |
| `document` | the update of the working document after each answer; the document is kept, but not updated nor sent to the model, when it is off |
| `reasoning` | the step-by-step reasoning shown before each answer, off by default because each step is a call to the model |
| `cache` | the offer to reuse the answer to a close question; the answers are kept, but not offered nor added, when it is off |
| `examples` | the answers kept as examples shown to the model; they are kept, and `/example add` still works, when it is off |
| `schedule` | the scheduled tasks and `/reports`; the tasks and the reports are kept when it is off |
| `prompt_versions` | the versions of the system prompt; the configured prompt is used, and the versions kept, when it is off |
| `logging` | the log of every message; the log is kept, and `python -m sensai stats` still reads it, when it is off |
| `fact_check` | the verification of each answer against its excerpts, off by default because each sentence is a call to `SENSAI_JUDGE_MODEL` |
| `sharing` | `/share conversation`, `/share document` and `python -m sensai publish`, off by default because anyone with a link can read its page; listing the links and taking them down always work |

The features are on by default, except `web_search`, `code_exec`, `files`, `reasoning`, `fact_check` and `sharing`. `SENSAI_TOOLS` and `SENSAI_WEB_SEARCH` became `SENSAI_FEATURE_TOOLS` and `SENSAI_FEATURE_WEB_SEARCH`: an `.env` that still uses them stops Sensai with the new name, instead of being ignored.

To add a feature, register it in `FEATURES` in `sensai/core/config.py`:

```python
FEATURES = {
    "rag": Feature("answer from the indexed documents"),
    "tools": Feature("let the model call the tools enabled in the configuration"),
    "web_search": Feature("let the model search the web; the queries are sent to Ollama's web search", default=False),
    "compaction": Feature("summarize older turns when the context fills up"),
}
```

Then check `config.features["compaction"]` where the feature runs. `SENSAI_FEATURE_COMPACTION`, `/features` and `/feature compaction off` work without other changes.

### Code layout

`sensai/core/` holds what Sensai does, and `sensai/interfaces/` how it is used: the CLI and the web page read the same events of `Agent.run`. The CLI is split by what each file is for:

| In `sensai/interfaces/cli/` | What it holds |
|---|---|
| `main.py` | the arguments of `python -m sensai`, and which run each one starts |
| `chat.py` | the chat loop |
| `startup.py` | what a session loads: the model, the tools, the rules, the index, the memory |
| `session.py` | the `Session`, what is asked at the keyboard, and saving the conversation |
| `answer.py` | an answer in the terminal, from the events of the agent |
| `output.py` | errors, warnings and the small text formats |
| `commands/` | the `/commands` of the chat, by subject; `dispatch.py` holds `/help` and sends each command to its function |
| `runs/` | the runs without a chat: `ingest` and `status`, `stats`, `eval`, `attack`, `schedule` |

A new `/command` goes in the file of its subject, with a line in `HELP` and in `handle_command` in `commands/dispatch.py`. The imports go one way, from `output.py` to `main.py`, so that any file can be imported first.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake Ollama client, Ollama does not need to run.

## Get involved

You're invited to join this project ! Check out the [contributing guide](./CONTRIBUTING.md).

If you're interested in how the project is organized at a higher level, please contact the current project manager.

## Our team ❤️

Developers
| [<img src="https://github.com/Manmohit509.png?size=85" width=85><br><sub>Manmohit Singh Lal</sub>](https://github.com/Manmohit509) | [<img src="https://github.com/AlexandreRsll.png?size=85" width=85><br><sub>Alexandre Rouselle</sub>](https://github.com/AlexandreRsll) | [<img src="https://github.com/edouardsilvestre.png?size=85" width=85><br><sub>Edouard Silvestre</sub>](https://github.com/edouardsilvestre)
| :---: | :---: | :---: |


<h2 align=center>
Support Us
</h2>


> 🚀 Don't hesitate to follow us on our different networks, and put a star 🌟 on the repositories
