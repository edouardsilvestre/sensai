# Sensai

**Sensai is an offline AI copilot for investigative journalists.**

A journalist receives a leak of 5,000 internal documents from a company. She can't upload a single file to the cloud without putting her source at risk. Sensai runs entirely on an air-gapped machine: it indexes the documents, identifies the people, companies and money flows involved, builds a timeline of events, and flags contradictions between documents.

Everything stays local, because protecting sources is non-negotiable. And every claim Sensai makes is tied back to the exact document and passage it comes from: in investigative journalism, an unsourced statement is a defamation risk. Faithfulness to sources isn't a nice-to-have here, it's the core of the product.

### Key features

- 🔒 **Fully offline by default**: no network access, no cloud, no data ever leaves the machine. Web search is opt-in, and every query is shown before it is sent.
- 💾 **Sessions & profile**: every conversation is saved and can be resumed days later, and a profile of standing instructions (the case, the language, what must never be written) is given to the model at the start of every conversation.
- 🌐 **Web search**: when enabled, the model can look up public facts (companies, people, news) and cites the URLs it used.
- 🔎 **Advanced RAG**: semantic and keyword search across thousands of heterogeneous documents (PDFs, emails, spreadsheets...), with every answer cited to its source.
- 🕸️ **Entity graph memory**: a structured knowledge graph of people, companies, accounts and their relationships: who is linked to whom, and how.
- 📅 **Timeline & contradiction detection**: events are placed on a chronology, and inconsistencies between documents are surfaced automatically.
- 📂 **Living investigation file**: a case document that grows as the investigation progresses, gathering findings, leads and their supporting evidence.
- 📊 **Code execution for financial analysis**: Sensai writes and runs code to analyze spreadsheets and trace money flows.
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

**Code execution** (`tools/code_runner.py`) lets the model run the Python code it writes, like ChatGPT's Code Interpreter. Each call starts a new Docker container from `SENSAI_SANDBOX_IMAGE`, sends the code to `python -` on its standard input and removes the container afterwards. The container has no network, 256 MB of memory, one CPU, 64 processes at most, a read-only disk (only `/tmp` is writable) and runs as `nobody` without privileges; nothing from the host is mounted. A program running longer than `SENSAI_CODE_TIMEOUT` seconds is killed. The model gets the exit code, the output and the traceback, each cut to their last 2000 characters. A crash is not a tool error: the model reads the traceback, fixes its code and calls `run_python` again, within the `SENSAI_MAX_TOOL_ROUNDS` limit. At startup Sensai checks that Docker runs and downloads the image if needed, otherwise it warns and chats without code execution.

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
| `/features` | list the features and whether they are on |
| `/feature NAME on\|off` | turn a feature on or off for the session |
| `/personas` | list the personas, `*` marks the active one |
| `/persona [ID]` | show the active persona, or switch to another one; the history is kept |
| `/sessions` | list the saved conversations, the current one marked with `*` |
| `/resume [ID]` | continue a saved conversation, the last one without ID |
| `/new` | start a new conversation; the current one stays saved |
| `/profile [add TEXT\|remove N]` | show, add or remove the instructions sent to the model about you |
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

With `SENSAI_FEATURE_WEB_SEARCH=true` (or `/feature web_search on` during a chat), the model searches the web when the question needs it. Each query is printed before it is sent, so you know exactly what leaves the machine:

```
> Quelle est la dernière version stable de Python publiée ? Cherche sur le web.
[web_search] query="la dernière version stable de Python"
[web_search] 5 results: www.python.org, devguide.python.org, docs.python.org
La dernière version stable de Python publiée est Python 3.15.1 [...]

URL de référence : https://www.python.org/downloads/latest/
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
| `SENSAI_FEATURE_SESSIONS` | `true` | save every conversation and allow resuming it |
| `SENSAI_FEATURE_PROFILE` | `true` | send the profile to the model |
| `SENSAI_COMPACT_THRESHOLD` | `75` | percentage of `SENSAI_NUM_CTX` at which the oldest turns are condensed (10-95) |
| `SENSAI_KEEP_RECENT_TURNS` | `4` | most recent turns always kept verbatim |
| `SENSAI_SESSIONS_DIR` | `.sensai/sessions` | folder where conversations are saved |
| `SENSAI_PROFILE_PATH` | `.sensai/profile.md` | your profile, one instruction per line |
| `SENSAI_EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding model; changing it requires `ingest --rebuild` |
| `SENSAI_INDEX_PATH` | `.sensai/index.db` | SQLite file holding the index |
| `SENSAI_RAG_TOP_K` | `5` | excerpts added to the prompt for each message |
| `SENSAI_RAG_CHUNK_SIZE` | `1200` | maximum chunk size, in characters |
| `SENSAI_RAG_CHUNK_OVERLAP` | `200` | characters repeated between consecutive chunks |
| `OLLAMA_API_KEY` | empty | key from [ollama.com/settings/keys](https://ollama.com/settings/keys), required by the web search |
| `SENSAI_WEB_SEARCH_RESULTS` | `5` | results the model gets for each web search (1-10) |
| `SENSAI_CODE_TIMEOUT` | `10` | seconds a program may run before it is stopped (1-60) |
| `SENSAI_SANDBOX_IMAGE` | `python:3.12-slim` | Docker image the code runs in; use one with pandas to analyse CSV files |
| `SENSAI_MAX_TOOL_ROUNDS` | `5` | rounds of tool calls allowed per message before the model has to answer (1-20) |

### Features

Each feature can be turned on or off with a `SENSAI_FEATURE_<NAME>` variable (`true` or `false`), or during a chat with `/feature NAME on|off`. A misspelled feature variable stops Sensai with an error.

| Feature | What it turns on or off |
|---|---|
| `rag` | answers drawn from the indexed documents |
| `tools` | every tool the model can call, whatever their own feature or setting |
| `web_search` | the web search tool, off by default because the queries leave the machine; needs `OLLAMA_API_KEY` |
| `code_exec` | the Python code execution tool, off by default because the code runs on this machine; needs Docker |

The features are on by default, except `web_search` and `code_exec`. `SENSAI_TOOLS` and `SENSAI_WEB_SEARCH` became `SENSAI_FEATURE_TOOLS` and `SENSAI_FEATURE_WEB_SEARCH`: an `.env` that still uses them stops Sensai with the new name, instead of being ignored.

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
