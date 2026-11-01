# Sensai

**Sensai is an offline AI copilot for investigative journalists.**

A journalist receives a leak of 5,000 internal documents from a company. She can't upload a single file to the cloud without putting her source at risk. Sensai runs entirely on an air-gapped machine: it indexes the documents, identifies the people, companies and money flows involved, builds a timeline of events, and flags contradictions between documents.

Everything stays local, because protecting sources is non-negotiable. And every claim Sensai makes is tied back to the exact document and passage it comes from: in investigative journalism, an unsourced statement is a defamation risk. Faithfulness to sources isn't a nice-to-have here, it's the core of the product.

### Key features

- 🔒 **Fully offline by default**: no network access, no cloud, no data ever leaves the machine. Web search is opt-in, and every query is shown before it is sent.
- 💾 **Sessions & profile**: every conversation is saved and can be resumed days later, and a profile of standing instructions (the case, the language, what must never be written) is given to the model at the start of every conversation.
- 🛡️ **Source & privacy guardrails**: the name of the source is removed from every answer and web search, personal data never leaves the machine, and requests targeting someone's private life are refused, following rules set in a file.
- 🌐 **Web search**: when enabled, the model can look up public facts (companies, people, news) and cites the URLs it used.
- 🔌 **MCP servers**: the model can call the tools of any MCP server reached by its URL, like a company registry lookup, each call approved first.
- 🔎 **Advanced RAG**: semantic and keyword search across thousands of heterogeneous documents (PDFs, emails, spreadsheets...), with every answer cited to its source.
- 🕸️ **Entity graph memory**: the model saves what is established about people, companies and places, and who is linked to whom, in a local database it reads and corrects across sessions.
- 📅 **Timeline & contradiction detection**: events are placed on a chronology, and inconsistencies between documents are surfaced automatically.
- 📂 **Living investigation file**: a case document, apart from the chat, that the model updates section by section after every answer with the facts established and their source, and that can be undone and exported to Markdown.
- 📊 **Code execution for financial analysis**: Sensai writes and runs code to analyze spreadsheets and trace money flows.
- 🧾 **Structured outputs**: `/json` answers in JSON that follows a fixed schema (the answer, its claims with their excerpts, the people and companies, the events), ready for a database or another tool.
- 🧠 **Traceable reasoning**: when turned on, Sensai reasons step by step (what it knows, what is missing, which document or tool to check next) and every step can be read again with `/reasoning`, so a conclusion can be checked before it is published.
- ♻️ **Semantic cache**: a question close to one already answered, in the same context, can get the same answer instantly once you confirm it, without calling the model again.
- 💡 **Answer examples**: keep your best answers with `/example add`, and Sensai shows the closest ones to the model for each new question, so that even a small model cites its excerpts and says what the documents do not say.
- 🗂️ **Prompt versioning**: each change to a persona prompt is kept as a dated version that can be compared, measured on your own test questions, and rolled back with `/prompt use N`.
- 🛡️ **Hallucination detection**: each generated statement is checked against the source documents; anything unsupported is flagged instead of presented as fact.
## How does it work?

`sensai/core` runs the conversation and emits events, `sensai/interfaces` displays them. Only `sensai/core/llm/ollama.py` talks to Ollama.

### Document search (RAG)

Indexing (`python -m sensai ingest`):

1. `rag/loaders.py` extracts the text of PDF, eml, csv, tsv, xlsx, html, md and txt files, with page numbers, sheet and row numbers, and email headers.
2. `rag/chunker.py` splits the text into sentences or rows and groups them into chunks of about 1200 characters, with 200 characters of overlap.
3. `rag/embedder.py` computes one vector per chunk with Ollama (`/api/embed`).
4. `rag/store.py` saves documents, chunks and vectors in one SQLite file, the vectors in a [sqlite-vec](https://github.com/asg017/sqlite-vec) table.

Files that did not change are skipped on the next run, and deleted files are removed from the index.

Answering: the message is embedded, the closest chunks are added to the prompt as numbered excerpts, and the model cites them as `[1]`, `[2]`. Only the message is kept in the history, not the excerpts.

### Sessions & profile

- `memory/sessions.py`: `SessionStore` writes each conversation to its own JSON file in `SENSAI_SESSIONS_DIR` after every answered message: the summary of older turns and the turns kept verbatim, with the sources each answer cited, exactly as `ConversationMemory` holds them. Resuming puts them back in the memory, so the model gets the same context as before the restart. A file is written next to the session then renamed, so a crash never leaves half a conversation, and an unreadable file is skipped instead of stopping Sensai.
- `memory/profile.py`: `Profile` reads `SENSAI_PROFILE_PATH`, a plain text file with one instruction per line, that `/profile add` and `/profile remove` edit and that can also be edited by hand. `ConversationMemory` sends it in the system message, after the system prompt and before the summary, so every conversation starts with it, new or resumed, and it is never condensed by the compaction.

Sessions and the profile are two features (`SENSAI_FEATURE_SESSIONS`, `SENSAI_FEATURE_PROFILE`). Like the index, they stay in `.sensai/` on the machine: delete `.sensai/sessions/` to erase every saved conversation.

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

### Prompt versions

With the `prompt_versions` feature (on by default), the system prompt of each persona is kept at every start in `SENSAI_PROMPTS_DIR/<id>.json`: a persona file edited since the last start becomes a new dated version, and the version the agent uses from then on.

```
Prompt of editor changed: kept as version 2. /prompts editor to list them.
[editor] > /prompts
Versions of the editor prompt (* in use):
  * 2. 2026-11-01 14:30  9c1e04ab  262 chars
    1. 2026-10-12 09:12  60f1efc2  248 chars
[editor] > /prompt diff 1 2
[editor] > /prompt use 1
editor uses version 1 of its prompt from now on.
```

`/prompt show N` prints a version, `/prompt diff N M` what changed between two, and `/prompt use N` rolls back to version N: the agent and the agents of `/review` use it from then on, and after a restart too, until the persona file changes again. Each saved conversation records the persona and the version of its prompt (`persona` and `prompt_version` in the session file), so an answer can be traced back to the prompt that wrote it.

`/prompt compare N M` asks the questions of `SENSAI_PROMPT_CASES_PATH` (see `prompt_cases.example.json`) with each version, and prints how many answers pass their checks, the average time and the average tokens per answer:

```
Version  Passed   Avg time  Avg tokens
1        2/3      4.12      412
2        3/3      3.05      298
  version 1, case 1 (Écris un titre : Marc Vidal a signé...): missing "480 000"
```

A case is a `question`, and optionally the texts its answer must contain (`must_contain`) and must not contain (`must_not_contain`), ignoring the case. To compare only the prompts, each question is asked alone with the system prompt, without the history, the documents nor the tools, at temperature 0 with a fixed seed. The result of each comparison is kept with the versions, and `/prompts` shows the last one. Turning the feature off goes back to the prompts of the persona files, and keeps the versions.

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

`/reasoning` lists the questions of the conversation, `/reasoning N` shows the thoughts, the tool calls and the full results of question N, so that the journalist can check what a conclusion rests on. The reasoning is never sent back to the model: only the question and the answer are kept in the history. `/new` and `/resume` clear it.

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

`examples.py`: `ExampleLibrary` keeps the examples in `SENSAI_EXAMPLES_PATH`, each with its question, its answer and the excerpts the answer cites, under their number so that `[2]` keeps its meaning. For each message, the question is embedded with `SENSAI_EMBED_MODEL` and compared by cosine similarity to the questions kept, and the `SENSAI_EXAMPLES_K` closest ones (2 by default) above a similarity of 0.5 are sent in a system message just before the question, with the instruction to answer the same way without reusing their facts. They never go to the history, and the step-by-step reasoning gets them too. Only plain answers can be kept: not `/json` nor `/review`, not an answer stopped by a guardrail, and never the same answer twice. Sensai ships no examples: only the journalist decides what a good answer is, and their examples stay on the machine, since they hold the names of the case.

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

Like the excerpts, tool calls and results are only sent for the current message: the history keeps the question and the final answer.

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

A rule looks for one thing and does one of three actions, `block`, `redact` (replaced by `[NAME]` or its `replacement`) or `warn`:

| Matcher | Finds | Example |
|---|---|---|
| `terms` | words or names, whatever their case, accents and spaces, as whole words | the source's name and codename |
| `pattern` | a regular expression (case-insensitive) | `adresse personnelle` |
| `detectors` | personal data, with their checksum: `email`, `phone`, `iban` (mod 97), `card` (Luhn), `nir` (French social security key), `ip` | `["iban", "card"]` |
| `judge` | messages that break a policy, judged by the local chat model (input only, block only) | "finding the home address of a private person" |

The judge is the chat model itself, so nothing leaves the machine, at the cost of one more call per message. It is asked to answer `ALLOW` or `BLOCK: reason`, and anything else blocks the message: the guardrails fail closed, like an invalid rules file, which stops Sensai at startup instead of running without its rules.

Without a rules file, one rule applies: personal data is blocked in web searches. A file at `SENSAI_GUARDRAILS_PATH` replaces it; [guardrails.example.json](guardrails.example.json) protects a source, keeps personal data out of web searches, flags bank details in answers and refuses requests about private life. Like the index, keep the file in `.sensai/`: it holds the names you protect.

## Getting Started

### Installation

Requirements: Python 3.10+ and a running [Ollama](https://ollama.com/download) server.

```bash
python3 -m venv .venv
source .venv/bin/activate
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

python -m sensai ingest ~/leak/          # index a folder (or files) for document search
python -m sensai ingest --rebuild ~/leak/   # start over, e.g. after changing the embedding model
python -m sensai status                  # what the index contains
python -m sensai -p fact_checker         # start with another persona
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
| `/sources` | show the full excerpts retrieved for the last message |
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
| `/profile [add TEXT\|remove N]` | show, add or remove the instructions sent to the model about you |
| `/document` | show the working document |
| `/document undo` | undo the changes made to the document by the last answer |
| `/document export PATH` | write the document to a Markdown file |
| `/json QUESTION` | answer in JSON that follows a schema: the answer, its claims with their excerpts, the entities and the events |
| `/guardrails` | list the rules checking the messages, the answers and the tool calls |
| `/approval [MODE]` | show when Sensai asks before running a tool, or change it: `critical`, `all` or `off` |
| `/reasoning [N]` | list the questions reasoned about in this conversation, or show the steps of question N |
| `/cache [clear]` | show how many answers the cache keeps and how often they were reused, or remove them all |
| `/examples` | list the answers kept as examples, `*` marks those shown to the model for the last answer |
| `/example add\|remove N` | keep the last answer as an example for similar questions, or remove example N |
| `/prompts [PERSONA]` | list the versions of a persona prompt (the active persona without PERSONA), `*` marks the one in use |
| `/prompt show\|use N [PERSONA]` | print version N of a persona prompt, or use it from now on (roll back) |
| `/prompt diff N M [PERSONA]` | show what changed between versions N and M |
| `/prompt compare N M [PERSONA]` | ask the test cases with versions N and M, and compare the checks passed, the time and the tokens |
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
| `SENSAI_NUM_CTX` | `8192` | context window of the chat model, in tokens |
| `SENSAI_THINK` | model default | `false` disables the thinking phase of models like qwen3.5 |
| `SENSAI_FEATURE_RAG` | `true` | answer from the indexed documents |
| `SENSAI_FEATURE_TOOLS` | `true` | `false` disables every tool, whatever their own setting |
| `SENSAI_FEATURE_WEB_SEARCH` | `false` | `true` lets the model search the web with Ollama's web search API; needs a model that supports tools and `OLLAMA_API_KEY` |
| `SENSAI_FEATURE_CODE_EXEC` | `false` | `true` lets the model run Python code in a Docker container; needs Docker and a model that supports tools |
| `SENSAI_FEATURE_FILES` | `false` | `true` lets the model list, read and write files in the allowed folders; needs a model that supports tools |
| `SENSAI_FEATURE_MEMORY` | `true` | let the model save and look up facts about people, companies and places; needs a model that supports tools |
| `SENSAI_FEATURE_MCP` | `true` | let the model call the tools of the MCP servers listed in `SENSAI_MCP_PATH`; needs a model that supports tools |
| `SENSAI_FEATURE_SESSIONS` | `true` | save every conversation and allow resuming it |
| `SENSAI_FEATURE_PROFILE` | `true` | send the profile to the model |
| `SENSAI_FEATURE_DOCUMENT` | `true` | update the working document after every answer, with one more call to the model |
| `SENSAI_FEATURE_CACHE` | `true` | offer the answer to a close question already asked, in the same context |
| `SENSAI_FEATURE_EXAMPLES` | `true` | show the model the answers kept as examples, for the questions close to theirs |
| `SENSAI_FEATURE_PROMPT_VERSIONS` | `true` | keep every version of the persona prompts, and use the one chosen with `/prompt use N` |
| `SENSAI_COMPACT_THRESHOLD` | `75` | percentage of `SENSAI_NUM_CTX` at which the oldest turns are condensed (10-95) |
| `SENSAI_KEEP_RECENT_TURNS` | `4` | most recent turns always kept verbatim |
| `SENSAI_SESSIONS_DIR` | `.sensai/sessions` | folder where conversations are saved |
| `SENSAI_PROFILE_PATH` | `.sensai/profile.md` | your profile, one instruction per line |
| `SENSAI_MEMORY_PATH` | `.sensai/memory.db` | SQLite file holding the long-term memory |
| `SENSAI_CACHE_PATH` | `.sensai/cache.db` | SQLite file holding the answers kept by the cache |
| `SENSAI_CACHE_THRESHOLD` | `90` | similarity, in percent, from which a kept answer is offered (50-100) |
| `SENSAI_FEATURE_GUARDRAILS` | `true` | check the messages, the answers and the tool calls against the rules |
| `SENSAI_COMPACT_THRESHOLD` | `75` | percentage of `SENSAI_NUM_CTX` at which the oldest turns are condensed (10-95) |
| `SENSAI_KEEP_RECENT_TURNS` | `4` | most recent turns always kept verbatim |
| `SENSAI_GUARDRAILS_PATH` | `.sensai/guardrails.json` | guardrail rules (see `guardrails.example.json`); without it, personal data is blocked in web searches |
| `SENSAI_MCP_PATH` | `.sensai/mcp.json` | URLs of the MCP servers (see `mcp.example.json`); without it, no server is reached |
| `SENSAI_EXAMPLES_PATH` | `.sensai/examples.json` | the answers kept as examples with `/example add` |
| `SENSAI_EXAMPLES_K` | `2` | examples shown to the model for each question, the closest ones first (0-5) |
| `SENSAI_PROMPTS_DIR` | `.sensai/prompts` | the versions of each persona prompt and the comparisons run, one file per persona |
| `SENSAI_PROMPT_CASES_PATH` | `.sensai/prompt_cases.json` | the test questions of `/prompt compare` (see `prompt_cases.example.json`) |
| `SENSAI_EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding model; changing it requires `ingest --rebuild` |
| `SENSAI_INDEX_PATH` | `.sensai/index.db` | SQLite file holding the index |
| `SENSAI_RAG_TOP_K` | `5` | excerpts added to the prompt for each message |
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

### Features

Each feature can be turned on or off with a `SENSAI_FEATURE_<NAME>` variable (`true` or `false`), or during a chat with `/feature NAME on|off`. A misspelled feature variable stops Sensai with an error.

| Feature | What it turns on or off |
|---|---|
| `rag` | answers drawn from the indexed documents |
| `tools` | every tool the model can call, whatever their own feature or setting |
| `web_search` | the web search tool, off by default because the queries leave the machine; needs `OLLAMA_API_KEY` |
| `code_exec` | the Python code execution tool, off by default because the code runs on this machine; needs Docker |
| `files` | the file tools, off by default because the model reads and writes files on this machine |
| `memory` | the long-term memory tools; the memory file is kept when it is off |
| `document` | the update of the working document after each answer; the document is kept, but not updated nor sent to the model, when it is off |
| `reasoning` | the step-by-step reasoning shown before each answer, off by default because each step is a call to the model |
| `cache` | the offer to reuse the answer to a close question; the answers are kept, but not offered nor added, when it is off |
| `examples` | the answers kept as examples shown to the model; they are kept, and `/example add` still works, when it is off |
| `prompt_versions` | the versions of the persona prompts; the persona files are used, and the versions kept, when it is off |

The features are on by default, except `web_search`, `code_exec`, `files` and `reasoning`. `SENSAI_TOOLS` and `SENSAI_WEB_SEARCH` became `SENSAI_FEATURE_TOOLS` and `SENSAI_FEATURE_WEB_SEARCH`: an `.env` that still uses them stops Sensai with the new name, instead of being ignored.

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
